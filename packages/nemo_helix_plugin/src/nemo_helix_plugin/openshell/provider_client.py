# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Typed provider-management operations over grpc.aio.

OpenShell 0.1.6 ships typed messages but its generated RPC methods have no
request/response annotations. Keep that untyped boundary here. This adapter
borrows the caller's async channel and adds no retries. Create, use and close
the channel on the same event loop; the caller owns its authentication and lifetime.
"""

from grpc.aio import Channel, UnaryUnaryMultiCallable
from openshell._proto import openshell_pb2 as pb
from openshell._proto.openshell_pb2_grpc import OpenShellStub


class ProviderGatewayClient:
    def __init__(self, channel: Channel, *, timeout: float = 30.0) -> None:
        stub = OpenShellStub(channel)
        self._timeout = timeout
        self._get_sandbox: UnaryUnaryMultiCallable[pb.GetSandboxRequest, pb.SandboxResponse] = stub.GetSandbox
        self._get_provider_profile: UnaryUnaryMultiCallable[
            pb.GetProviderProfileRequest, pb.ProviderProfileResponse
        ] = stub.GetProviderProfile
        self._get_provider: UnaryUnaryMultiCallable[pb.GetProviderRequest, pb.ProviderResponse] = stub.GetProvider
        self._create_provider: UnaryUnaryMultiCallable[pb.CreateProviderRequest, pb.ProviderResponse] = (
            stub.CreateProvider
        )
        self._update_provider: UnaryUnaryMultiCallable[pb.UpdateProviderRequest, pb.ProviderResponse] = (
            stub.UpdateProvider
        )
        self._detach_sandbox_provider: UnaryUnaryMultiCallable[
            pb.DetachSandboxProviderRequest, pb.DetachSandboxProviderResponse
        ] = stub.DetachSandboxProvider
        self._delete_provider: UnaryUnaryMultiCallable[pb.DeleteProviderRequest, pb.DeleteProviderResponse] = (
            stub.DeleteProvider
        )
        self._list_providers: UnaryUnaryMultiCallable[pb.ListProvidersRequest, pb.ListProvidersResponse] = (
            stub.ListProviders
        )

    async def get_sandbox(self, request: pb.GetSandboxRequest) -> pb.SandboxResponse:
        return await self._get_sandbox(request, timeout=self._timeout)

    async def get_provider_profile(self, request: pb.GetProviderProfileRequest) -> pb.ProviderProfileResponse:
        return await self._get_provider_profile(request, timeout=self._timeout)

    async def get_provider(self, request: pb.GetProviderRequest) -> pb.ProviderResponse:
        return await self._get_provider(request, timeout=self._timeout)

    async def create_provider(self, request: pb.CreateProviderRequest) -> pb.ProviderResponse:
        return await self._create_provider(request, timeout=self._timeout)

    async def update_provider(self, request: pb.UpdateProviderRequest) -> pb.ProviderResponse:
        return await self._update_provider(request, timeout=self._timeout)

    async def detach_sandbox_provider(
        self, request: pb.DetachSandboxProviderRequest
    ) -> pb.DetachSandboxProviderResponse:
        return await self._detach_sandbox_provider(request, timeout=self._timeout)

    async def delete_provider(self, request: pb.DeleteProviderRequest) -> pb.DeleteProviderResponse:
        return await self._delete_provider(request, timeout=self._timeout)

    async def list_providers(self, request: pb.ListProvidersRequest) -> pb.ListProvidersResponse:
        return await self._list_providers(request, timeout=self._timeout)
