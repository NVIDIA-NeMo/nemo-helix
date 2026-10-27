# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Typed provider-management operations over the SDK's synchronous gRPC stub.

OpenShell 0.1.6 ships typed messages but its generated RPC methods have no
request/response annotations. Keep that untyped boundary here. This adapter
reuses the caller's authenticated channel, never owns it, and adds no retries.
"""

import asyncio

from openshell._proto import openshell_pb2 as pb
from openshell._proto.openshell_pb2_grpc import OpenShellStub


class ProviderGatewayClient:
    def __init__(self, stub: OpenShellStub, *, timeout: float = 30.0) -> None:
        self._stub = stub
        self._timeout = timeout

    async def get_sandbox(self, request: pb.GetSandboxRequest) -> pb.SandboxResponse:
        return await asyncio.to_thread(self._stub.GetSandbox, request, timeout=self._timeout)

    async def get_provider_profile(self, request: pb.GetProviderProfileRequest) -> pb.ProviderProfileResponse:
        return await asyncio.to_thread(self._stub.GetProviderProfile, request, timeout=self._timeout)

    async def get_provider(self, request: pb.GetProviderRequest) -> pb.ProviderResponse:
        return await asyncio.to_thread(self._stub.GetProvider, request, timeout=self._timeout)

    async def create_provider(self, request: pb.CreateProviderRequest) -> pb.ProviderResponse:
        return await asyncio.to_thread(self._stub.CreateProvider, request, timeout=self._timeout)

    async def update_provider(self, request: pb.UpdateProviderRequest) -> pb.ProviderResponse:
        return await asyncio.to_thread(self._stub.UpdateProvider, request, timeout=self._timeout)

    async def detach_sandbox_provider(
        self, request: pb.DetachSandboxProviderRequest
    ) -> pb.DetachSandboxProviderResponse:
        return await asyncio.to_thread(self._stub.DetachSandboxProvider, request, timeout=self._timeout)

    async def delete_provider(self, request: pb.DeleteProviderRequest) -> pb.DeleteProviderResponse:
        return await asyncio.to_thread(self._stub.DeleteProvider, request, timeout=self._timeout)

    async def list_providers(self, request: pb.ListProvidersRequest) -> pb.ListProvidersResponse:
        return await asyncio.to_thread(self._stub.ListProviders, request, timeout=self._timeout)
