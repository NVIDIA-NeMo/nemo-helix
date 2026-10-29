# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Serve OpenShell credential requests using secrets stored in Helix."""

from __future__ import annotations

import asyncio
import time
from collections.abc import Callable
from datetime import datetime, timezone
from typing import Any

import grpc
from google.protobuf.message import Message
from nemo_deployments_plugin.entities import Deployment
from nemo_helix_plugin.client.errors import NemoHTTPError, NemoTransportError
from nemo_helix_plugin.entities.client import AsyncEntitiesClient
from nemo_helix_plugin.entity_client import NemoEntitiesClient, NemoEntityNotFoundError
from nemo_helix_plugin.iam.client import AsyncIAMClient
from nemo_helix_plugin.iam.types import AuthzRequest
from nemo_helix_plugin.jobs.client import AsyncJobsClient
from nemo_helix_plugin.openshell.types import GatewayWorkload
from nemo_helix_plugin.secrets.client import AsyncSecretsClient
from openshell._proto import credential_driver_pb2_grpc as driver_grpc  # ty: ignore[unresolved-import]
from pydantic import BaseModel, ConfigDict, Field, SecretStr, ValidationError
from sqlalchemy.exc import SQLAlchemyError

from . import wire
from .registry import BindingRegistry, ProviderOwner
from .tickets import RegistrationDenied

DRIVER_NAME = "nemo"
CAPABILITY = "openshell.credentials.contract"


class Handle(BaseModel):
    model_config = ConfigDict(extra="forbid")
    driver: str = DRIVER_NAME
    handle: str = ""
    metadata: dict[str, str] = Field(default_factory=dict)

    def identifier(self) -> str:
        if self.driver != DRIVER_NAME or not self.handle or self.metadata:
            raise RegistrationDenied("Invalid credential handle.")
        return self.handle


class StoreRequest(ProviderOwner):
    value: SecretStr
    existing_handle: Handle | None = None
    object_id: str = ""


class LookupRequest(ProviderOwner):
    handle: Handle
    request_id: str = ""


class ResolveRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    credentials: list[LookupRequest] = Field(min_length=1, max_length=32)


class CredentialDriver(driver_grpc.CredentialDriverServicer):
    """Let OpenShell providers use Helix secrets through the credential-driver protocol.

    StoreCredential accepts a signed registration request and returns a handle to
    its stored secret reference. ResolveCredentials checks the binding, workload
    and creator access, then reads the value using the driver's Helix identity.
    Auth-disabled installations omit the creator permission check.
    DeleteCredential revokes the handle without deleting the underlying secret.
    """

    def __init__(
        self,
        registry: BindingRegistry,
        secrets_client: AsyncSecretsClient,
        *,
        cache_lifetime: int = 300,
        auth_enabled: bool = True,
        clock: Callable[[], float] = time.time,
    ) -> None:
        if not 1 <= cache_lifetime <= 3600:
            raise ValueError("Credential cache lifetime must be between 1 and 3600 seconds.")
        self.registry = registry
        self.secrets_client = secrets_client
        self.cache_lifetime = cache_lifetime
        self.auth_enabled = auth_enabled
        self.clock = clock

    def _require_auth_mode(self, principal_id: str | None) -> None:
        if self.auth_enabled != (principal_id is not None):
            raise RegistrationDenied("Binding authentication mode does not match this driver.")

    async def _require_creator_access(self, principal_id: str | None, workspace: str) -> None:
        if principal_id is None:
            return  # Explicit auth-disabled bindings do not query the PDP.
        if principal_id.startswith("service:"):
            raise RegistrationDenied("A binding requires a user or service-account creator.")
        response = await AsyncIAMClient.from_client(self.secrets_client).evaluate_authorization(
            entrypoint="has_permissions",
            body=AuthzRequest(
                input={
                    "principal_id": principal_id,
                    "workspace": workspace,
                    "permissions": ["secrets.read"],
                }
            ),
        )
        if response.data().result.get("allowed") is not True:
            raise RegistrationDenied("The workload creator cannot use this secret.")

    async def _require_active_workload(self, workload: GatewayWorkload | None) -> None:
        if workload is None:
            return  # Standalone tickets retain their bounded binding lifetime.
        if workload.kind == "job":
            if not workload.attempt_id or not workload.step_name:
                raise RegistrationDenied("Missing job execution identity.")
            jobs = AsyncJobsClient.from_client(self.secrets_client)
            job = (await jobs.get_job(workspace=workload.workspace, name=workload.name)).data()
            step = (
                await jobs.get_job_step(workspace=workload.workspace, job=workload.name, name=workload.step_name)
            ).data()
            active = {"created", "pending", "active"}
            if (
                step.id != workload.id
                or job.attempt_id != workload.attempt_id
                or step.attempt_id != workload.attempt_id
                or job.status not in active
                or step.status not in active
            ):
                raise RegistrationDenied("Job execution is no longer active.")
        else:
            entities = NemoEntitiesClient(AsyncEntitiesClient.from_client(self.secrets_client))
            try:
                deployment = await entities.get(Deployment, workload.name, workspace=workload.workspace)
            except NemoEntityNotFoundError:
                raise RegistrationDenied("Deployment no longer exists.") from None
            if (
                deployment.id != workload.id
                or deployment.desired_state != "READY"
                or deployment.status
                not in {
                    "PENDING",
                    "STARTING",
                    "READY",
                    "LOST",
                }
            ):
                raise RegistrationDenied("Deployment is no longer active.")

    async def dispatch(self, method: str, request: Message, context: grpc.aio.ServicerContext) -> Message:
        try:
            return await self._dispatch(method, wire.data(request))
        except (RegistrationDenied, ValidationError):
            await context.abort(grpc.StatusCode.PERMISSION_DENIED, "Credential binding request denied.")
        except NemoHTTPError as exc:
            code = (
                grpc.StatusCode.PERMISSION_DENIED if exc.status_code in (401, 403, 404) else grpc.StatusCode.UNAVAILABLE
            )
            await context.abort(code, "NeMo secret access failed.")
        except (NemoTransportError, SQLAlchemyError, OSError, TimeoutError):
            await context.abort(grpc.StatusCode.UNAVAILABLE, "Credential backend unavailable.")
        except Exception:
            # Do not let gRPC expose request values, token-bearing HTTP errors,
            # database paths, or exception reprs to the gateway.
            await context.abort(grpc.StatusCode.INTERNAL, "Credential driver failed.")
        raise RuntimeError("gRPC abort did not terminate the request")

    async def _dispatch(self, method: str, raw: dict[str, Any]) -> Message:
        if method == "GetCapabilities":
            gateway = raw.get("gateway", {})
            if (
                gateway.get("protocol_version", {}).get("major") != 1
                or CAPABILITY not in gateway.get("supported_capabilities", [])
                or set(gateway.get("required_capabilities", [])) - {CAPABILITY}
            ):
                raise RegistrationDenied("Incompatible gateway credential protocol.")
            return wire.message(
                "GetCredentialDriverCapabilitiesResponse",
                {
                    "driver_name": DRIVER_NAME,
                    "backend_kind": "nemo-secrets",
                    "supports_expires_at": True,
                    "extension": {
                        "protocol_version": {"major": 1},
                        "implementation_name": "nemo-helix/credential-driver",
                        "implementation_version": "experimental-v1",
                        "supported_capabilities": [CAPABILITY],
                        "required_capabilities": [CAPABILITY],
                    },
                },
            )
        if method == "StoreCredential":
            request = StoreRequest.model_validate(raw)
            self._require_auth_mode(self.registry.tickets.verify(request.value).principal_id)
            owner = ProviderOwner.model_validate(request.model_dump(include=ProviderOwner.model_fields.keys()))
            handle = await asyncio.to_thread(
                self.registry.register,
                owner,
                request.value,
                object_id=request.object_id,
                existing_handle=request.existing_handle.identifier() if request.existing_handle else "",
            )
            return wire.message("StoreCredentialResponse", {"handle": {"driver": DRIVER_NAME, "handle": handle}})
        if method == "DeleteCredential":
            request = LookupRequest.model_validate(raw)
            owner = ProviderOwner.model_validate(request.model_dump(include=ProviderOwner.model_fields.keys()))
            await asyncio.to_thread(self.registry.delete, owner, request.handle.identifier())
            return wire.message("DeleteCredentialResponse")
        if method == "ResolveCredentials":
            requests = ResolveRequest.model_validate(raw).credentials
            ids = [item.request_id for item in requests]
            if not all(ids) or len(set(ids)) != len(ids):
                raise RegistrationDenied("Invalid batch request identities.")
            # Validate every binding before making any Secrets API calls.
            bindings = [
                await asyncio.to_thread(
                    self.registry.resolve,
                    ProviderOwner.model_validate(item.model_dump(include=ProviderOwner.model_fields.keys())),
                    item.handle.identifier(),
                )
                for item in requests
            ]
            results = []
            deadlines = []
            for binding in bindings:
                self._require_auth_mode(binding.claims.principal_id)
            for item, binding in zip(requests, bindings, strict=True):
                # A slow later read must not extend the lease of an earlier
                # value in the batch. Start its lifetime before remote access.
                deadline = int(self.clock()) + self.cache_lifetime
                if binding.claims.workload is None:
                    deadline = min(deadline, binding.claims.binding_expires_at)
                deadlines.append(deadline)
                # The bearer token authenticates the driver. The signed creator
                # is checked explicitly by the PDP, never through identity headers.
                async with asyncio.timeout(15):
                    await self._require_active_workload(binding.claims.workload)
                    await self._require_creator_access(binding.claims.principal_id, binding.claims.secret_workspace)
                    response = await self.secrets_client.access_secret(
                        workspace=binding.claims.secret_workspace,
                        name=binding.claims.secret_name,
                    )
                secret = response.data()
                if (secret.workspace, secret.name) != (binding.claims.secret_workspace, binding.claims.secret_name):
                    raise RegistrationDenied("Secret response scope mismatch.")
                results.append({"request_id": item.request_id, "value": secret.value})
            # A revoke/expiry while the remote read was pending must win.
            for result, binding, deadline in zip(results, bindings, deadlines, strict=True):
                await asyncio.to_thread(self.registry.resolve, binding.owner, binding.handle)
                async with asyncio.timeout(15):
                    await self._require_active_workload(binding.claims.workload)
                result["expiration_time"] = datetime.fromtimestamp(deadline, tz=timezone.utc).isoformat()
            if any(deadline <= int(self.clock()) for deadline in deadlines):
                raise RegistrationDenied("Credential resolution exceeded its cache lifetime.")
            return wire.message("ResolveCredentialsResponse", {"credentials": results})
        raise RegistrationDenied("Unsupported credential operation.")

    async def GetCapabilities(self, request: Message, context: grpc.aio.ServicerContext) -> Message:
        return await self.dispatch("GetCapabilities", request, context)

    async def StoreCredential(self, request: Message, context: grpc.aio.ServicerContext) -> Message:
        return await self.dispatch("StoreCredential", request, context)

    async def ResolveCredentials(self, request: Message, context: grpc.aio.ServicerContext) -> Message:
        return await self.dispatch("ResolveCredentials", request, context)

    async def DeleteCredential(self, request: Message, context: grpc.aio.ServicerContext) -> Message:
        return await self.dispatch("DeleteCredential", request, context)

    async def ListCredentials(self, request: Message, context: grpc.aio.ServicerContext) -> None:
        await context.abort(grpc.StatusCode.UNIMPLEMENTED, "Credential enumeration is not supported.")

    def add_to_server(self, server: grpc.aio.Server) -> None:
        driver_grpc.add_CredentialDriverServicer_to_server(self, server)
