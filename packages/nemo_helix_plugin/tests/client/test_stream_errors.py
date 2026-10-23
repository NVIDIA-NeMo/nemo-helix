# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from contextlib import asynccontextmanager, contextmanager

import httpx
import pytest
from nemo_helix_plugin.client.errors import UnprocessableEntityError
from nemo_helix_plugin.client.response import AsyncNemoStreamResponse, NemoStreamResponse
from nemo_helix_plugin.client.types import PreparedRequest
from pydantic import BaseModel


class Frame(BaseModel):
    text: str


def request() -> PreparedRequest:
    return PreparedRequest(
        path_template="/preview",
        path_params={},
        method="POST",
        content=None,
        content_type=None,
        response_type=Frame,
    )


@pytest.mark.parametrize("status", [200, 422])
def test_sync_stream_preserves_error_body_without_buffering_success(status: int) -> None:
    raw = httpx.Response(
        status,
        stream=httpx.ByteStream(b'{"detail":"Unknown model provider"}'),
        headers={"content-type": "application/json"},
    )

    @contextmanager
    def stream():
        try:
            yield raw
        finally:
            raw.close()

    response = NemoStreamResponse(stream(), Frame, request())
    if status == 422:
        with pytest.raises(UnprocessableEntityError, match="Unknown model provider") as exc:
            with response.stream():
                pytest.fail("Error must be raised before yielding frames")
        assert exc.value.body == {"detail": "Unknown model provider"}
    else:
        with response.stream():
            assert not raw.is_stream_consumed


@pytest.mark.asyncio
@pytest.mark.parametrize("status", [200, 422])
async def test_async_stream_preserves_error_body_without_buffering_success(status: int) -> None:
    raw = httpx.Response(
        status,
        stream=httpx.ByteStream(b'{"detail":"Unknown model provider"}'),
        headers={"content-type": "application/json"},
    )

    @asynccontextmanager
    async def stream():
        try:
            yield raw
        finally:
            await raw.aclose()

    response = AsyncNemoStreamResponse(stream(), Frame, request())
    if status == 422:
        with pytest.raises(UnprocessableEntityError, match="Unknown model provider") as exc:
            async with response.stream():
                pytest.fail("Error must be raised before yielding frames")
        assert exc.value.body == {"detail": "Unknown model provider"}
    else:
        async with response.stream():
            assert not raw.is_stream_consumed
