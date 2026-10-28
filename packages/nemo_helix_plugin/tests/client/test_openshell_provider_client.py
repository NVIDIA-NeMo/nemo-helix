# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Async transport, deadline and cancellation behavior against a real gRPC server."""

# ruff: noqa: E402 -- Check the optional SDK before imports.

import asyncio
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

import pytest

pytest.importorskip("openshell")

import grpc
from nemo_helix_plugin.openshell.provider_client import ProviderGatewayClient
from openshell._proto import datamodel_pb2 as dm
from openshell._proto import openshell_pb2 as pb
from openshell._proto import openshell_pb2_grpc as rpc


class Gateway(rpc.OpenShellServicer):
    def __init__(self) -> None:
        self.started = asyncio.Event()
        self.cancelled = asyncio.Event()

    async def GetProvider(
        self, request: pb.GetProviderRequest, context: grpc.aio.ServicerContext
    ) -> pb.ProviderResponse:
        if request.name == "missing":
            await context.abort(grpc.StatusCode.NOT_FOUND, "No provider")
        if request.name == "blocked":
            self.started.set()
            try:
                await asyncio.Future()
            finally:
                self.cancelled.set()
        return pb.ProviderResponse(provider=dm.Provider(metadata=dm.ObjectMeta(name=request.name)))


@asynccontextmanager
async def gateway() -> AsyncGenerator[tuple[Gateway, grpc.aio.Channel]]:
    service = Gateway()
    server = grpc.aio.server()
    rpc.add_OpenShellServicer_to_server(service, server)
    port = server.add_insecure_port("127.0.0.1:0")
    await server.start()
    try:
        async with grpc.aio.insecure_channel(f"127.0.0.1:{port}") as channel:
            yield service, channel
    finally:
        await server.stop(None)


@pytest.mark.asyncio
async def test_cancellation_reaches_gateway_without_blocking_other_calls() -> None:
    async with gateway() as (service, channel):
        client = ProviderGatewayClient(channel)
        pending = asyncio.create_task(client.get_provider(pb.GetProviderRequest(name="blocked")))
        try:
            await asyncio.wait_for(service.started.wait(), timeout=2)
            response = await asyncio.wait_for(client.get_provider(pb.GetProviderRequest(name="ready")), timeout=2)
            assert response.provider.metadata.name == "ready"
        finally:
            pending.cancel()
            with pytest.raises(asyncio.CancelledError):
                await pending
        await asyncio.wait_for(service.cancelled.wait(), timeout=2)
        # Cancelling one RPC must leave the caller-owned channel usable.
        response = await client.get_provider(pb.GetProviderRequest(name="still-ready"))
        assert response.provider.metadata.name == "still-ready"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("name", "expected"), [("missing", grpc.StatusCode.NOT_FOUND), ("blocked", grpc.StatusCode.DEADLINE_EXCEEDED)]
)
async def test_status_and_configured_deadline_are_preserved(name: str, expected: grpc.StatusCode) -> None:
    async with gateway() as (_, channel):
        client = ProviderGatewayClient(channel, timeout=0.2)
        with pytest.raises(grpc.aio.AioRpcError) as error:
            await client.get_provider(pb.GetProviderRequest(name=name))
        assert error.value.code() == expected
