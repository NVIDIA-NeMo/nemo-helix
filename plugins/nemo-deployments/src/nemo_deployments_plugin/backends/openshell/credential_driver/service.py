# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""OpenShell gRPC adapter. Stores references, resolves via the typed NeMo client."""

from __future__ import annotations

import asyncio
import time
from collections.abc import Callable
from datetime import datetime, timezone
from typing import Any

import grpc
from google.protobuf.message import Message
from nemo_helix_plugin.client.errors import NemoHTTPError, NemoTransportError
from nemo_helix_plugin.secrets.client import AsyncSecretsClient
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


class CredentialDriver:
    def __init__(
        self,
        registry: BindingRegistry,
        secrets_client: AsyncSecretsClient,
        *,
        cache_lifetime: int = 30,
        clock: Callable[[], float] = time.time,
    ) -> None:
        if not 1 <= cache_lifetime <= 60:
            raise ValueError("Credential cache lifetime must be between 1 and 60 seconds.")
        self.registry = registry
        self.secrets_client = secrets_client
        self.cache_lifetime = cache_lifetime
        self.clock = clock

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
            owner = ProviderOwner.model_validate(request.model_dump(include=ProviderOwner.model_fields.keys()))
            handle = self.registry.register(
                owner,
                request.value,
                object_id=request.object_id,
                existing_handle=request.existing_handle.identifier() if request.existing_handle else "",
            )
            return wire.message("StoreCredentialResponse", {"handle": {"driver": DRIVER_NAME, "handle": handle}})
        if method == "DeleteCredential":
            request = LookupRequest.model_validate(raw)
            owner = ProviderOwner.model_validate(request.model_dump(include=ProviderOwner.model_fields.keys()))
            self.registry.delete(owner, request.handle.identifier())
            return wire.message("DeleteCredentialResponse")
        if method == "ResolveCredentials":
            requests = ResolveRequest.model_validate(raw).credentials
            ids = [item.request_id for item in requests]
            if not all(ids) or len(set(ids)) != len(ids):
                raise RegistrationDenied("Invalid batch request identities.")
            # Validate every binding before making any Secrets API calls.
            bindings = [
                self.registry.resolve(
                    ProviderOwner.model_validate(item.model_dump(include=ProviderOwner.model_fields.keys())),
                    item.handle.identifier(),
                )
                for item in requests
            ]
            results = []
            for item, binding in zip(requests, bindings, strict=True):
                async with asyncio.timeout(15):
                    response = await self.secrets_client.access_secret(
                        workspace=binding.claims.secret_workspace,
                        name=binding.claims.secret_name,
                    )
                secret = response.data()
                if (secret.workspace, secret.name) != (binding.claims.secret_workspace, binding.claims.secret_name):
                    raise RegistrationDenied("Secret response scope mismatch.")
                results.append({"request_id": item.request_id, "value": secret.value})
            # A revoke/expiry while the remote read was pending must win.
            now = int(self.clock())
            for result, binding in zip(results, bindings, strict=True):
                self.registry.resolve(binding.owner, binding.handle)
                deadline = min(now + self.cache_lifetime, binding.claims.binding_expires_at)
                result["expiration_time"] = datetime.fromtimestamp(deadline, tz=timezone.utc).isoformat()
            return wire.message("ResolveCredentialsResponse", {"credentials": results})
        raise RegistrationDenied("Unsupported credential operation.")

    def add_to_server(self, server: grpc.aio.Server) -> None:
        handlers = {}
        for name, (request_name, _) in wire.METHODS.items():

            def handler_for(method: str) -> Callable:
                async def handle(request: Message, context: grpc.aio.ServicerContext) -> Message:
                    if method == "ListCredentials":
                        await context.abort(grpc.StatusCode.UNIMPLEMENTED, "Credential enumeration is not supported.")
                    return await self.dispatch(method, request, context)

                return handle

            handlers[name] = grpc.unary_unary_rpc_method_handler(
                handler_for(name),
                request_deserializer=wire.message_type(request_name).FromString,
                response_serializer=lambda response: response.SerializeToString(),
            )
        server.add_generic_rpc_handlers((grpc.method_handlers_generic_handler(wire.SERVICE, handlers),))
