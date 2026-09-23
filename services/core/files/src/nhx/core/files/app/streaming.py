# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Streaming file upload utilities."""

import logging
from abc import ABC, abstractmethod
from contextlib import asynccontextmanager, nullcontext
from typing import AsyncIterator, TypeVar

import aiohttp
import anyio
from anyio import to_thread
from anyio.streams.memory import MemoryObjectSendStream
from fastapi import Request
from fastapi.datastructures import Headers
from nhx.common.files.storage_config import DEFAULT_READ_CHUNK_SIZE
from nhx.core.files.app.backends.base import ByteRange
from nhx.core.files.exceptions import InactivityTimeoutError
from starlette.requests import ClientDisconnect
from streaming_form_data import StreamingFormDataParser
from streaming_form_data.targets import BaseTarget
from yarl import URL

logger = logging.getLogger(__name__)


DEFAULT_QUEUE_SIZE = 10
# 10s balances responsiveness with tolerance for transient network slowness
DEFAULT_INACTIVITY_TIMEOUT_SECONDS = 10
DEFAULT_MULTIPART_BUFFER_SIZE = 1 * 1024 * 1024  # 1MB batches to reduce parser calls


class ChunkProcessor(ABC):
    @abstractmethod
    async def process(self, chunk: bytes, send_stream: MemoryObjectSendStream[bytes]) -> None: ...

    @abstractmethod
    async def process_remaining_buffer(self, send_stream: MemoryObjectSendStream[bytes]) -> None: ...


class OctetStreamChunkProcessor(ChunkProcessor):
    async def process(self, chunk: bytes, send_stream: MemoryObjectSendStream[bytes]):
        await send_stream.send(chunk)

    async def process_remaining_buffer(self, send_stream: MemoryObjectSendStream[bytes]) -> None:
        # We don't buffer any bytes, so no need to flush.
        return


class MultipartChunkProcessor(ChunkProcessor):
    def __init__(
        self,
        headers: Headers,
        filepath: str | None = None,
        buffer_size: int = DEFAULT_MULTIPART_BUFFER_SIZE,
    ) -> None:
        self.parser = StreamingFormDataParser(headers=headers)

        # In the future, we can make this configurable based upon `purpose`
        self.target = StreamingTarget()
        self.parser.register("file", self.target)

        self.buffer = bytearray()
        self.buffer_size = buffer_size

    async def process(self, chunk: bytes, send_stream: MemoryObjectSendStream[bytes]):
        self.buffer.extend(chunk)
        # Only parse when self.buffer reaches threshold
        if len(self.buffer) >= self.buffer_size:
            await to_thread.run_sync(self.parser.data_received, bytes(self.buffer))
            self.buffer.clear()

            # Pull whatever data accumulated and send to stream
            if data := self.target.get_and_clear_buffer():
                await send_stream.send(data)

    async def process_remaining_buffer(self, send_stream: MemoryObjectSendStream[bytes]):
        # Parse any remaining data
        if self.buffer:
            await to_thread.run_sync(self.parser.data_received, bytes(self.buffer))
            if data := self.target.get_and_clear_buffer():
                await send_stream.send(data)


@asynccontextmanager
async def streaming_file_upload(
    request: Request,
    chunk_processor: ChunkProcessor,
    queue_size: int = DEFAULT_QUEUE_SIZE,
    inactivity_timeout_seconds: float = DEFAULT_INACTIVITY_TIMEOUT_SECONDS,
) -> AsyncIterator[AsyncIterator[bytes]]:
    """
    A context manager for streaming file uploads without spooling to disk.

    The rate at which the request bytes come in likely won't match the rate
    at which a reader can read the bytes. To decouple these two tasks,
    we create a producer/consumer pattern using anyio object streams. We spin up
    a background task that'll take bytes off of the request and put it into
    the stream for consumption.

    Args:
        request: FastAPI request object
        chunk_processor: ChunkProcessor instance to handle chunk processing
        queue_size: Size of the memory buffer queue
        inactivity_timeout_seconds: Timeout in seconds for receiving data. If no bytes
            are received within this period, the connection will be terminated with
            InactivityTimeoutError.

    Usage:
        >>> async with streaming_file_upload(request, OctetStreamChunkProcessor()) as upload:
        ...     async for chunk in upload:
        ...         await file.write(chunk)
    """
    send_stream, receive_stream = anyio.create_memory_object_stream[bytes](max_buffer_size=queue_size)

    async def _parse_and_send() -> None:
        """Background task that reads from request stream and sends data with timeout monitoring."""
        async with send_stream:
            try:
                async for raw_chunk in await iter_with_inactivity_timeout(
                    request.stream(),
                    inactivity_timeout_seconds,
                ):
                    await chunk_processor.process(raw_chunk, send_stream)
                await chunk_processor.process_remaining_buffer(send_stream)

            except ClientDisconnect:
                logger.warning(f"Client disconnected during upload for {request.url.path}")
                raise
            except InactivityTimeoutError:
                logger.warning(
                    f"No bytes received within {inactivity_timeout_seconds} seconds for {request.url.path}, closing request."
                )
                raise
            except anyio.BrokenResourceError:
                # Expected when consumer exits early and closes receive_stream
                logger.debug(f"Consumer closed early for {request.url.path}")
            except Exception:
                logger.exception(f"Error parsing and sending multipart data for {request.url.path}")
                raise

    try:
        async with anyio.create_task_group() as tg:
            tg.start_soon(_parse_and_send)
            async with receive_stream:
                yield receive_stream
    except ExceptionGroup as eg:
        # Tasks started in TaskGroups will raise as ExceptionGroups,
        # so we normalize them to regular exceptions here.
        match eg.exceptions:
            case [single_exception]:
                raise single_exception
            case _:
                raise


class StreamingTarget(BaseTarget):
    """
    A streaming-form-data target that accumulates file chunks in a buffer.

    The async consumer pulls all accumulated data from the buffer after each
    parser.data_received() call completes.
    """

    def __init__(self):
        super().__init__()
        self._buffer = bytearray()

    def on_data_received(self, chunk: bytes) -> None:
        """Called when a chunk of file data is received."""
        if chunk:
            self._buffer.extend(chunk)

    def get_and_clear_buffer(self) -> bytes:
        """Get all buffered data and clear the buffer."""
        if self._buffer:
            data = bytes(self._buffer)
            self._buffer.clear()
            return data
        return b""

    @property
    def filename(self) -> str | None:
        """Get the filename from multipart headers if this is a file upload."""
        # The multipart_filename property is set by streaming-form-data
        return getattr(self, "multipart_filename", None)


T = TypeVar("T")


async def iter_with_inactivity_timeout(
    content: AsyncIterator[T],
    timeout_seconds: float | None = DEFAULT_INACTIVITY_TIMEOUT_SECONDS,
    preflight: bool = False,
) -> AsyncIterator[T]:
    """
    Wrap an async iterator with inactivity timeout and optional preflight check.

    When preflight=True, reads the first chunk immediately during the await call,
    surfacing errors before the caller commits to a StreamingResponse. This allows
    callers to catch connection errors and return proper HTTP error responses.

    Uses a CancelScope around each source read (not across ``yield``), so the
    inactivity deadline applies only while waiting on the upstream iterator.
    Yielding inside a CancelScope is unsafe: closing the async generator on
    early consumer exit can exit the scope from another task.

    Args:
        content: The async iterator to wrap
        timeout_seconds: Max time to wait for each chunk (inactivity timeout).
            Pass None to disable timeout.
        preflight: If True, read first chunk immediately (during await) to surface
            errors before returning the iterator. Use this when downloading from
            remote sources (HF, NGC, S3) where connection errors are common.
            Not needed for uploads (receiving from client) or when already inside
            an aiohttp context that surfaces errors naturally.

    Returns:
        An async iterator that yields items with inactivity timeout applied.

    Raises:
        InactivityTimeoutError: If no data received within timeout period
    """
    first_chunk: T | None = None

    if preflight:
        # Read first chunk NOW (during the await) to surface errors early
        try:
            with anyio.fail_after(timeout_seconds):
                first_chunk = await anext(content, None)
        except TimeoutError as e:
            raise InactivityTimeoutError(f"No data received within {timeout_seconds} seconds") from e
        # If first_chunk is None (empty iterator), _stream() handles it naturally:
        # the loop will immediately get StopAsyncIteration and exit

    async def _stream() -> AsyncIterator[T]:
        # Yield preflight chunk if we have one
        if first_chunk is not None:
            yield first_chunk

        # Await under fail_after, then yield outside the CancelScope. Spanning a
        # yield with a CancelScope breaks when the consumer exits early and the
        # async generator is aclosed from another task.
        while True:
            try:
                with anyio.fail_after(timeout_seconds):
                    item = await anext(content)
            except StopAsyncIteration:
                break
            except TimeoutError as e:
                raise InactivityTimeoutError(f"No data received within {timeout_seconds} seconds") from e
            yield item

    return _stream()


@asynccontextmanager
async def tee_stream(
    input_stream: AsyncIterator[bytes],
    queue_size: int = DEFAULT_QUEUE_SIZE,
    timeout_seconds: float | None = DEFAULT_INACTIVITY_TIMEOUT_SECONDS,
) -> AsyncIterator[tuple[AsyncIterator[bytes], AsyncIterator[bytes]]]:
    """
    Split an input stream into two independent output streams (like Unix tee).

    This allows you to consume the same data from two different places simultaneously,
    such as writing to disk while streaming to a client.

    Each output stream has its own buffer queue, so consumers can read at different
    rates. If one consumer is slower, it won't block the other (up to queue_size).

    Args:
        input_stream: The input async iterator to split
        queue_size: Size of each output stream's buffer queue
        timeout_seconds: Timeout for sending to each stream. If a consumer
            is not reading fast enough, InactivityTimeoutError is raised. None to disable.

    Usage:
        >>> async with tee_stream(upload_stream) as (disk_stream, response_stream):
        ...     async with anyio.create_task_group() as tg:
        ...         # Background task writes to disk
        ...         async def write_to_disk():
        ...             async for chunk in disk_stream:
        ...                 await file.write(chunk)
        ...         tg.start_soon(write_to_disk)
        ...
        ...         # Main flow streams to client (runs concurrently with write_to_disk)
        ...         async for chunk in response_stream:
        ...             yield chunk
        ...
        ...         # Task group ensures disk write completes before exiting

    Yields:
        Tuple of two independent async iterators

    Raises:
        InactivityTimeoutError: If a consumer doesn't read within timeout_seconds.
    """
    send_stream1, receive_stream1 = anyio.create_memory_object_stream[bytes](max_buffer_size=queue_size)
    send_stream2, receive_stream2 = anyio.create_memory_object_stream[bytes](max_buffer_size=queue_size)

    async def _send_with_timeout(send_stream, chunk: bytes):
        """Send chunk to stream with timeout."""
        try:
            if timeout_seconds is None:
                await send_stream.send(chunk)
            else:
                with anyio.fail_after(timeout_seconds):
                    await send_stream.send(chunk)
        except TimeoutError as e:
            raise InactivityTimeoutError(f"Consumer not reading fast enough; blocked for {timeout_seconds}s)") from e

    async def _distribute():
        """Read from input and write to both outputs."""
        async with send_stream1, send_stream2:
            try:
                async for chunk in input_stream:
                    await _send_with_timeout(send_stream1, chunk)
                    await _send_with_timeout(send_stream2, chunk)
            except anyio.BrokenResourceError:
                # Expected if consumers close early
                logger.debug("Consumer closed early during tee")
            except Exception:
                logger.exception("Error distributing stream")
                raise

    try:
        async with anyio.create_task_group() as tg:
            tg.start_soon(_distribute)
            async with receive_stream1, receive_stream2:
                yield receive_stream1, receive_stream2
    except ExceptionGroup as eg:
        # Normalize ExceptionGroup to single exception if possible
        match eg.exceptions:
            case [single_exception]:
                raise single_exception
            case _:
                raise


async def download_url(
    url: str,
    headers: dict | None = None,
) -> dict:
    """
    Fetch JSON data from a URL.

    Args:
        url: The URL to fetch
        headers: Optional HTTP headers

    Returns:
        The parsed JSON response as a dict

    Raises:
        aiohttp.ClientResponseError: For HTTP errors (4xx, 5xx)
        aiohttp.ClientError: For network/connection errors
    """
    if headers is None:
        headers = {}

    async with aiohttp.ClientSession() as session:
        # Use encoded=True to prevent aiohttp/yarl from re-encoding
        # already-percent-encoded query parameters in the URL.
        async with session.get(URL(url, encoded=True), headers=headers) as response:
            response.raise_for_status()
            return await response.json()


async def upload_url_streaming(
    url: str,
    data: AsyncIterator[bytes],
    headers: dict | None = None,
    session: aiohttp.ClientSession | None = None,
    method: str = "PUT",
) -> int:
    """
    Upload streaming data to a URL via HTTP request.

    Streams data directly from the async iterator to the HTTP request body,
    minimizing memory usage and maximizing throughput.

    Args:
        url: The URL to upload to (e.g., presigned S3 PUT URL)
        data: Async iterator of bytes to upload
        headers: Optional HTTP headers
        session: Optional aiohttp session to reuse
        method: HTTP method to use (default: PUT)

    Returns:
        The number of bytes uploaded (from Content-Length response header, or 0)

    Raises:
        aiohttp.ClientResponseError: For HTTP errors (4xx, 5xx)
        aiohttp.ClientError: For network/connection errors
    """
    if headers is None:
        headers = {}

    if session is None:
        session_cm = aiohttp.ClientSession()
    else:
        session_cm = nullcontext(session)

    # No timeout - uploads can take a long time
    request_timeout = aiohttp.ClientTimeout(total=None)

    async with session_cm as session:
        # Use encoded=True to prevent aiohttp/yarl from re-encoding
        # already-percent-encoded query parameters (e.g. signed URLs
        # whose cryptographic signatures are invalidated by re-encoding).
        async with session.request(
            method,
            URL(url, encoded=True),
            data=data,
            headers=headers,
            timeout=request_timeout,
        ) as response:
            response.raise_for_status()
            # Return content length if available
            return int(response.headers.get("Content-Length", 0))


async def download_url_streaming(
    url: str,
    session: aiohttp.ClientSession,
    headers: dict | None = None,
    byte_range: ByteRange | None = None,
    chunk_size: int = DEFAULT_READ_CHUNK_SIZE,
) -> AsyncIterator[bytes]:
    """Stream content from a URL.

    Args:
        url: The URL to download from
        session: aiohttp session to use for requests (caller manages lifetime)
        headers: Optional HTTP headers to include
        byte_range: Optional byte range for partial content requests
        chunk_size: Size of chunks to yield
    """
    if headers is None:
        headers = {}

    if byte_range is not None:
        headers["Range"] = f"bytes={byte_range.start}-{byte_range.end}"

    request_timeout = aiohttp.ClientTimeout(total=None)

    # Use encoded=True to prevent aiohttp/yarl from re-encoding
    # already-percent-encoded query parameters (e.g. signed URLs from
    # NGC whose cryptographic signatures are invalidated by re-encoding).
    async with session.get(URL(url, encoded=True), headers=headers, timeout=request_timeout) as response:
        response.raise_for_status()

        async for chunk in response.content.iter_chunked(chunk_size):
            yield chunk
