# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Manage OpenShell service access for deployments using credentials stored in Helix Secrets."""

from __future__ import annotations

import hashlib
import json
import os
import stat
import time
from collections.abc import Callable, Mapping
from pathlib import Path

import grpc
from nemo_helix_plugin.auth import AuthContext
from nemo_helix_plugin.client.errors import NotFoundError, PermissionDeniedError
from nemo_helix_plugin.entities.base import parse_qualified_name
from nemo_helix_plugin.openshell.provider_client import ProviderGatewayClient
from nemo_helix_plugin.openshell.registration import BindingScope, RegistrationTickets
from nemo_helix_plugin.openshell.types import (
    GatewayCredentialConfig,
    GatewayProviderBinding,
    GatewayWorkload,
    ProviderAccessRevoked,
)
from nemo_helix_plugin.secrets.client import AsyncSecretsClient
from openshell._proto import datamodel_pb2 as dm
from openshell._proto import openshell_pb2 as pb
from pydantic import JsonValue, SecretStr

from .authorization import authorize_secret_binding

_OWNER = "nemo-helix/provider-owner"
_BINDING = "nemo-helix/provider-binding"
_GATEWAY = "nemo-helix/gateway"
_SANDBOX = "nemo-helix/sandbox"
_CREATED = "nemo-helix/created"
_RENEWED = "nemo-helix/renewed"
_REVOKED = "nemo-helix/revoked"


class ProviderProvisioningError(ValueError):
    """A provider cannot be attached with the requested scope."""


def read_registration_key(path: Path) -> bytes:
    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    with os.fdopen(descriptor, "rb") as stream:
        info = os.fstat(stream.fileno())
        if not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid() or stat.S_IMODE(info.st_mode) & 0o077:
            raise ProviderProvisioningError("Registration key must be a private regular file owned by the controller.")
        key = stream.read(65537)
        if not 32 <= len(key) <= 65536:
            raise ProviderProvisioningError("Invalid registration key size.")
        return key


def _digest(value: JsonValue) -> str:
    # OpenShell label values are limited to 63 characters. Retain 252 bits of
    # the digest so ownership and binding fingerprints fit that API contract.
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()[:63]


def _rpc_code(error: grpc.RpcError) -> grpc.StatusCode | None:
    code = getattr(error, "code", None)
    status = code() if callable(code) else None
    return status if isinstance(status, grpc.StatusCode) else None


class DeploymentProviderProvisioner:
    """Manage the OpenShell providers used by a deployment's sandbox.

    The Deployments backend calls prepare to register or renew provider credentials,
    and cleanup after sandbox deletion. Registration sends signed secret references
    to the gateway; the Helix credential driver resolves the actual values.

    Provider labels identify the owning workload and credential configuration.
    Renewal timestamps and revocation markers live on the gateway so another
    controller can resume the work after a restart.
    """

    def __init__(
        self,
        config: GatewayCredentialConfig,
        workspace: str,
        channel: grpc.aio.Channel,
        *,
        timeout: float = 30.0,
        tickets: RegistrationTickets | None = None,
        clock: Callable[[], float] = time.time,
    ) -> None:
        """Borrow the backend's async channel; the backend owns its lifetime."""
        self.config = config
        self.workspace = workspace
        self.client = ProviderGatewayClient(channel, timeout=timeout)
        self.clock = clock
        self.tickets = tickets or RegistrationTickets(
            read_registration_key(config.registration_key_file), config.gateway_id
        )

    def _scope(self) -> dm.WorkspaceSelector:
        return dm.WorkspaceSelector(workspace=self.workspace)

    async def prepare(
        self,
        bindings: dict[str, GatewayProviderBinding],
        *,
        workload_workspace: str,
        workload_id: str,
        workload: GatewayWorkload,
        auth_context: AuthContext | None,
        secrets_client: AsyncSecretsClient,
        platform_host: str,
        renew: bool = False,
        auth_enabled: bool = True,
    ) -> list[str]:
        """Return provider names for the backend to attach, or renew attached providers.

        Initial calls create providers or reuse matching registrations. With
        renew=True, Helix-managed renewal rechecks access and replaces credentials
        using fresh signed requests; it never recreates a missing provider.

        Permission loss during renewal records revocation before detaching and
        deleting the affected provider. This removes service access, not the
        sandbox process; the caller receives ProviderAccessRevoked.
        """
        if workload.kind != "deployment":
            raise ProviderProvisioningError("Expected a deployment credential binding.")
        if renew and self.config.renewal_mode != "helix":
            return []
        if not self.config.credential_refresh_verified:
            raise ProviderProvisioningError(
                "Gateway credential refresh must be verified before enabling provider workloads."
            )
        principal: str | None = None
        if auth_enabled:
            if auth_context is None:
                raise ProviderProvisioningError("Gateway providers require the workload creator's auth context.")
            principal = auth_context.principal_on_behalf_of or auth_context.principal_id
            if not principal or principal.startswith("service:"):
                raise ProviderProvisioningError("Gateway providers require a user or service-account creator.")
        owner = _digest([self.config.gateway_id, workload_workspace, workload_id])
        sandbox: pb.Sandbox | None = None
        if renew:
            sandbox = (
                await self.client.get_sandbox(pb.GetSandboxRequest(workspace_scope=self._scope(), name=workload_id))
            ).sandbox
            if sandbox.status.phase not in (pb.SANDBOX_PHASE_PROVISIONING, pb.SANDBOX_PHASE_READY):
                return []
        names: list[str] = []
        revoked = False
        keys: set[str] = set()
        # Validate every profile before registering anything.
        for binding in bindings.values():
            approved = self.config.profiles.get(binding.profile)
            if approved is None or workload_workspace not in approved.workspaces:
                raise ProviderProvisioningError("Injection profile is not approved for this workspace.")
            if approved.profile_workspace not in ("", self.workspace):
                raise ProviderProvisioningError("Profile workspace must be global or match the gateway workspace.")
            if set(binding.credentials) != approved.credential_keys or keys.intersection(binding.credentials):
                raise ProviderProvisioningError(
                    "Provider credential slots must match the approved profile and be unique."
                )
            keys.update(binding.credentials)
            response = await self.client.get_provider_profile(
                pb.GetProviderProfileRequest(
                    workspace_scope=dm.WorkspaceSelector(workspace=approved.profile_workspace)
                    if approved.profile_workspace
                    else None,
                    id=binding.profile,
                )
            )
            profile = response.profile
            if not profile.endpoints or any(
                not endpoint.host or "*" in endpoint.host or endpoint.host.lower() == platform_host.lower()
                for endpoint in profile.endpoints
            ):
                raise ProviderProvisioningError("Provider profiles must target explicit services outside the platform.")
            if any(
                credential.HasField("token_grant")
                or (credential.HasField("refresh") and credential.refresh.strategy not in (0, 1, 2))
                for credential in profile.credentials
            ):
                raise ProviderProvisioningError("Gateway secret bindings require static credential profiles.")
            declared = {key for credential in profile.credentials for key in credential.env_vars}
            if not set(binding.credentials) <= declared:
                raise ProviderProvisioningError("Gateway profile does not declare the requested credential slots.")

        for alias, binding in sorted(bindings.items()):
            approved = self.config.profiles[binding.profile]
            fingerprint = _digest([binding.model_dump(), principal, approved.profile_workspace, workload.model_dump()])
            name = "nhx-" + _digest([owner, alias, workload.id, workload.attempt_id])[:40]
            labels: dict[str, str] = {
                _OWNER: owner,
                _BINDING: fingerprint,
                _GATEWAY: _digest(self.config.gateway_id),
                _SANDBOX: workload_id,
            }
            existing: pb.ProviderResponse | None
            try:
                existing = await self.client.get_provider(
                    pb.GetProviderRequest(workspace_scope=self._scope(), name=name)
                )
            except grpc.RpcError as exc:
                if _rpc_code(exc) != grpc.StatusCode.NOT_FOUND:
                    raise
                if renew:
                    # A completed cleanup leaves no provider. Never recreate it
                    # during renewal, even if permission has since been restored.
                    revoked = True
                    continue
                existing = None
            else:
                if (
                    any(existing.provider.metadata.labels.get(key) != value for key, value in labels.items())
                    or existing.provider.type != binding.profile
                    or existing.provider.profile_workspace != approved.profile_workspace
                ):
                    raise ProviderProvisioningError("Existing provider does not belong to this workload binding.")
                if existing.provider.config.get(_REVOKED):
                    await self._finish_revocation(name, workload_id)
                    revoked = True
                    continue
                if not renew:
                    names.append(name)
                    continue
                if sandbox is None or name not in sandbox.spec.providers:
                    raise ProviderProvisioningError("Cannot renew a provider detached from this sandbox.")
                try:
                    renewed = float(
                        existing.provider.config.get(_RENEWED, existing.provider.metadata.labels.get(_RENEWED, "0"))
                    )
                except ValueError:
                    renewed = 0
                age = self.clock() - renewed
                if 0 <= age < self.config.renewal_interval_seconds:
                    names.append(name)
                    continue
            credentials: dict[str, str] = {}
            try:
                for key, ref in binding.credentials.items():
                    secret_workspace, secret_name = parse_qualified_name(ref, default_workspace=workload_workspace)
                    ticket = await self._issue_ticket(
                        BindingScope(
                            gateway_id=self.config.gateway_id,
                            gateway_workspace=self.workspace,
                            provider=name,
                            credential_key=key,
                            secret_workspace=secret_workspace,
                            secret_name=secret_name,
                            job_id=workload_id,
                            principal_id=principal,
                            workload=workload,
                        ),
                        requester_secrets=secrets_client,
                    )
                    credentials[key] = ticket.get_secret_value()
            except (PermissionError, PermissionDeniedError, NotFoundError):
                if not renew or existing is None:
                    raise
                # Persist intent before detaching. A new controller can finish
                # cleanup even if permission returns after this observation.
                updated = dm.Provider()
                updated.CopyFrom(existing.provider)
                updated.credentials.clear()
                updated.credential_handles.clear()
                updated.config[_REVOKED] = "1"
                await self.client.update_provider(
                    pb.UpdateProviderRequest(workspace_scope=self._scope(), provider=updated)
                )
                await self._finish_revocation(name, workload_id)
                revoked = True
                continue
            provider = dm.Provider(
                metadata=dm.ObjectMeta(
                    name=name,
                    workspace=self.workspace,
                    labels={**labels, _CREATED: str(int(self.clock()))},
                ),
                type=binding.profile,
                profile_workspace=approved.profile_workspace,
                credentials=credentials,
                config={_RENEWED: str(self.clock())},
            )
            if renew:
                # Preserve the gateway's optimistic-concurrency version and
                # immutable provider identity. Never forward redacted values or
                # caller-supplied handles from GetProvider back to UpdateProvider.
                assert existing is not None
                updated = dm.Provider()
                updated.CopyFrom(existing.provider)
                updated.credentials.clear()
                updated.credential_handles.clear()
                updated.credentials.update(credentials)
                # Commit the renewal timestamp with the credentials so restarted
                # controllers observe the last successful update.
                updated.config[_RENEWED] = str(self.clock())
                await self.client.update_provider(
                    pb.UpdateProviderRequest(workspace_scope=self._scope(), provider=updated)
                )
                names.append(name)
                continue
            try:
                await self.client.create_provider(
                    pb.CreateProviderRequest(workspace_scope=self._scope(), provider=provider)
                )
            except grpc.RpcError as exc:
                if _rpc_code(exc) != grpc.StatusCode.ALREADY_EXISTS:
                    raise
                existing = await self.client.get_provider(
                    pb.GetProviderRequest(workspace_scope=self._scope(), name=name)
                )
                if (
                    any(existing.provider.metadata.labels.get(key) != value for key, value in labels.items())
                    or existing.provider.type != binding.profile
                    or existing.provider.profile_workspace != approved.profile_workspace
                ):
                    raise ProviderProvisioningError(
                        "Concurrent provider registration belongs to another binding."
                    ) from None
                if existing.provider.config.get(_REVOKED):
                    await self._finish_revocation(name, workload_id)
                    revoked = True
                    continue
            names.append(name)
        if revoked:
            raise ProviderAccessRevoked("Gateway provider access was removed. Start a new workload to restore access.")
        return names

    async def _issue_ticket(self, scope: BindingScope, *, requester_secrets: AsyncSecretsClient) -> SecretStr:
        """Check creator permission and secret metadata before signing the registration request."""
        await authorize_secret_binding(
            workspace=scope.secret_workspace,
            name=scope.secret_name,
            principal_id=scope.principal_id,
            secrets_client=requester_secrets,
        )
        return self.tickets.sign(scope)

    async def _finish_revocation(self, name: str, sandbox_name: str) -> None:
        """Detach the provider from the sandbox, then delete the provider from OpenShell."""
        try:
            sandbox = (
                await self.client.get_sandbox(pb.GetSandboxRequest(workspace_scope=self._scope(), name=sandbox_name))
            ).sandbox
        except grpc.RpcError as exc:
            if _rpc_code(exc) != grpc.StatusCode.NOT_FOUND:
                raise
        else:
            if name in sandbox.spec.providers:
                await self.client.detach_sandbox_provider(
                    pb.DetachSandboxProviderRequest(
                        workspace_scope=self._scope(),
                        sandbox=sandbox_name,
                        provider=name,
                        expected_resource_version=sandbox.metadata.resource_version,
                    )
                )
        await self._delete(name)

    async def cleanup(self, *, workload_workspace: str, workload_id: str) -> None:
        """Delete this workload's providers after the backend confirms sandbox deletion.

        Gateway ownership labels identify the providers even after a controller
        restart. The backend must retry failed cleanup on a later reconciliation.
        """
        owner = _digest([self.config.gateway_id, workload_workspace, workload_id])
        providers = await self._list_providers()
        for provider in providers:
            if provider.metadata.labels.get(_OWNER) == owner:
                await self._delete(provider.metadata.name)

    async def cleanup_unused(self) -> None:
        """Finish pending revocations and delete providers whose sandboxes are missing.

        Unmarked providers get a ten-minute grace period for sandbox creation.
        Revocation markers are processed immediately. OpenShell refuses deletion
        while a provider remains attached.
        """
        for provider in await self._list_providers():
            labels: Mapping[str, str] = provider.metadata.labels
            if labels.get(_GATEWAY) != _digest(self.config.gateway_id) or not labels.get(_SANDBOX):
                continue
            if provider.config.get(_REVOKED):
                await self._finish_revocation(provider.metadata.name, labels[_SANDBOX])
                continue
            try:
                created = int(labels.get(_CREATED, ""))
            except ValueError:
                continue
            if self.clock() - created < 600:
                continue
            try:
                await self.client.get_sandbox(
                    pb.GetSandboxRequest(workspace_scope=self._scope(), name=labels[_SANDBOX])
                )
            except grpc.RpcError as exc:
                if _rpc_code(exc) != grpc.StatusCode.NOT_FOUND:
                    raise
                await self._delete(provider.metadata.name)

    async def _delete(self, name: str) -> None:
        await self.client.delete_provider(
            pb.DeleteProviderRequest(workspace_scope=self._scope(), name=name, allow_missing=True)
        )

    async def _list_providers(self) -> list[dm.Provider]:
        providers: list[dm.Provider] = []
        token = ""
        seen: set[str] = set()
        while True:
            response = await self.client.list_providers(
                pb.ListProvidersRequest(workspace_scope=self._scope(), page_size=100, page_token=token)
            )
            providers.extend(response.providers)
            token = response.next_page_token
            if not token:
                return providers
            if token in seen:
                raise ProviderProvisioningError("Gateway returned a repeated provider page token.")
            seen.add(token)
