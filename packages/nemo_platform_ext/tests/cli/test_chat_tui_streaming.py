# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The SSE decoder and the typed binary stream ``nemo chat`` consumes."""

from __future__ import annotations

from contextlib import contextmanager
from unittest.mock import MagicMock

import click
import httpx
import pytest
from nemo_platform_ext.cli.chat_tui import (
    ServerSentEvent,
    SSEDecoder,
    _iter_stream_deltas,
)
from nemo_platform_plugin.client.response import NemoBinaryResponse


def _decode(*chunks: bytes) -> list[ServerSentEvent]:
    return list(SSEDecoder().iter_bytes(iter(chunks)))


def test_decoder_dispatches_on_blank_line_and_joins_multiline_data() -> None:
    events = _decode(b"event: ping\ndata: one\ndata: two\n\ndata: three\n\n")

    assert events == [
        ServerSentEvent(event="ping", data="one\ntwo"),
        ServerSentEvent(event=None, data="three"),
    ]


def test_decoder_handles_lines_split_across_chunks_and_crlf() -> None:
    events = _decode(b"data: hel", b"lo\r\n\r", b"\ndata: wor", b"ld\r\n\r\n")

    assert events == [ServerSentEvent(data="hello"), ServerSentEvent(data="world")]


def test_decoder_ignores_comments_and_strips_one_leading_space() -> None:
    events = _decode(b": keep-alive\ndata:  two spaces\n\n")

    assert events == [ServerSentEvent(data=" two spaces")]


def test_decoder_flushes_a_trailing_event_without_final_blank_line() -> None:
    assert _decode(b"data: [DONE]") == [ServerSentEvent(data="[DONE]")]


def _binary_response(body: bytes) -> NemoBinaryResponse:
    @contextmanager
    def stream_ctx():
        yield httpx.Response(200, stream=httpx.ByteStream(body), request=httpx.Request("POST", "http://test"))

    return NemoBinaryResponse(stream_ctx(), MagicMock())


def test_iter_stream_deltas_reads_a_typed_binary_response() -> None:
    body = b'data: {"choices": [{"delta": {"content": "hi"}}]}\n\ndata: [DONE]\n\n'

    deltas = list(_iter_stream_deltas(_binary_response(body)))

    assert deltas == [{"choices": [{"delta": {"content": "hi"}}]}]


def test_iter_stream_deltas_surfaces_error_events() -> None:
    body = b'event: error\ndata: {"detail": "upstream exploded"}\n\n'

    with pytest.raises(click.ClickException, match="upstream exploded"):
        list(_iter_stream_deltas(_binary_response(body)))
