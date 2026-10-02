# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for streaming file upload functionality."""

from unittest.mock import AsyncMock, MagicMock, Mock, patch

import anyio
import pytest
from fastapi import Request
from nhx.core.files.app.backends.base import ByteRange
from nhx.core.files.app.streaming import (
    OctetStreamChunkProcessor,
    download_url,
    download_url_streaming,
    iter_with_inactivity_timeout,
    streaming_file_upload,
    tee_stream,
)
from nhx.core.files.exceptions import InactivityTimeoutError
from starlette.requests import ClientDisconnect


def create_streaming_request(content: bytes, filename: str = "test.txt") -> Request:
    """Helper to create a mock Request with raw binary streaming content."""

    async def async_iter_body():
        """Simulate streaming request body in chunks."""
        chunk_size = 1024
        for i in range(0, len(content), chunk_size):
            yield content[i : i + chunk_size]

    mock_request = Mock(spec=Request)
    mock_request.headers = {"content-type": "application/octet-stream"}
    mock_request.stream = lambda: async_iter_body()
    mock_request.url = Mock()
    mock_request.url.path = "/test/path"

    return mock_request


def create_mock_aiohttp_session(mock_response, streaming=False):
    """Helper to create a mock aiohttp ClientSession with proper async context manager support.

    Args:
        mock_response: The mock response object to return from session.get()
        streaming: If True, use AsyncMock for session.get (for streaming downloads).
                   If False, use MagicMock (for non-streaming downloads with async with).
    """
    # Setup response as async context manager
    mock_response.__aenter__ = AsyncMock(return_value=mock_response)
    mock_response.__aexit__ = AsyncMock(return_value=None)

    # Create session mock
    mock_session = MagicMock()
    # For streaming: session.get needs to be awaited first (AsyncMock)
    # For non-streaming: session.get is used directly in async with (MagicMock)
    if streaming:
        mock_session.get = AsyncMock(return_value=mock_response)
    else:
        mock_session.get = MagicMock(return_value=mock_response)
    mock_session.__aenter__ = AsyncMock(return_value=mock_session)
    mock_session.__aexit__ = AsyncMock(return_value=None)

    return mock_session


async def test_upload_large_file():
    """Test uploading a large file that fills the queue."""
    content = b"x" * (1024 * 100)  # 100KB
    request = create_streaming_request(content)

    chunks = []
    async with streaming_file_upload(request, OctetStreamChunkProcessor()) as upload:
        async for chunk in upload:
            chunks.append(chunk)

    result = b"".join(chunks)
    assert result == content
    assert len(chunks) > 1  # Should be multiple chunks


async def test_upload_empty_file():
    """Test uploading an empty file."""
    content = b""
    request = create_streaming_request(content)

    chunks = []
    async with streaming_file_upload(request, OctetStreamChunkProcessor()) as upload:
        async for chunk in upload:
            chunks.append(chunk)

    result = b"".join(chunks)
    assert result == content


async def test_early_exit():
    """Test consumer exiting early (not reading all chunks)."""
    content = b"x" * (1024 * 100)  # 100KB
    request = create_streaming_request(content)

    chunks = []
    async with streaming_file_upload(request, OctetStreamChunkProcessor()) as upload:
        async for chunk in upload:
            chunks.append(chunk)
            # Exit early after first chunk
            if len(chunks) >= 1:
                break

    # Should complete without deadlock
    assert len(chunks) >= 1


async def test_exception_during_consumption():
    """Test that exceptions during consumption are handled properly."""
    content = b"x" * 1024
    request = create_streaming_request(content)

    class CustomException(Exception):
        pass

    # Exception will be wrapped in ExceptionGroup by task group
    with pytest.raises((CustomException, ExceptionGroup)) as exc_info:
        async with streaming_file_upload(request, OctetStreamChunkProcessor()) as upload:
            async for _ in upload:
                raise CustomException("Test exception")

    # Verify the original exception is present (either directly or in the group)
    if isinstance(exc_info.value, ExceptionGroup):
        assert any(isinstance(e, CustomException) for e in exc_info.value.exceptions)


async def test_concurrent_uploads():
    """Test multiple concurrent uploads don't interfere with each other."""
    content1 = b"Content 1: " + b"x" * 1000
    content2 = b"Content 2: " + b"y" * 1000

    request1 = create_streaming_request(content1, "file1.txt")
    request2 = create_streaming_request(content2, "file2.txt")

    results = {}

    async def upload_file(name, request):
        chunks = []
        async with streaming_file_upload(request, OctetStreamChunkProcessor()) as upload:
            async for chunk in upload:
                chunks.append(chunk)
        result = b"".join(chunks)
        results[name] = result

    async with anyio.create_task_group() as tg:
        tg.start_soon(upload_file, "file1", request1)
        tg.start_soon(upload_file, "file2", request2)

    assert results["file1"] == content1
    assert results["file2"] == content2


async def test_backpressure():
    """Test that backpressure works correctly with a small queue."""
    # Create content large enough to generate many chunks
    content = b"x" * (1024 * 50)  # 50KB
    request = create_streaming_request(content)

    # Use a very small queue to ensure backpressure is exercised
    queue_size = 2
    chunks = []

    async with streaming_file_upload(request, OctetStreamChunkProcessor(), queue_size=queue_size) as upload:
        async for chunk in upload:
            chunks.append(chunk)

    result = b"".join(chunks)
    assert result == content
    # If we receive more chunks than the queue size, it means the producer
    # had to wait for the consumer to drain the queue (backpressure worked)
    assert len(chunks) > queue_size, f"Expected more than {queue_size} chunks, got {len(chunks)}"


async def test_client_disconnect_during_upload():
    """Test that client disconnect during upload is handled properly."""
    content = b"x" * (1024 * 100)  # 100KB

    # Create a request that disconnects after sending some chunks
    disconnect_after_chunks = 10
    chunks_sent = 0

    async def async_iter_body_with_disconnect():
        """Simulate streaming request body that disconnects mid-stream."""
        nonlocal chunks_sent
        chunk_size = 1024
        for i in range(0, len(content), chunk_size):
            if chunks_sent >= disconnect_after_chunks:
                raise ClientDisconnect()
            yield content[i : i + chunk_size]
            chunks_sent += 1

    mock_request = Mock(spec=Request)
    mock_request.headers = {"content-type": "application/octet-stream"}
    mock_request.stream = lambda: async_iter_body_with_disconnect()
    mock_request.url = Mock()
    mock_request.url.path = "/test/disconnect"

    # The client disconnect should propagate as an exception
    with pytest.raises((ClientDisconnect, ExceptionGroup)) as exc_info:
        async with streaming_file_upload(mock_request, OctetStreamChunkProcessor()) as upload:
            async for _ in upload:
                pass  # Consumer tries to read all chunks

    # Verify the exception is either ClientDisconnect directly or wrapped in ExceptionGroup
    if isinstance(exc_info.value, ExceptionGroup):
        assert any(isinstance(e, ClientDisconnect) for e in exc_info.value.exceptions)


async def test_iter_with_inactivity_timeout_normal_flow():
    """Test iter_with_inactivity_timeout with normal data flow."""

    async def normal_iterator():
        """Iterator that yields data quickly."""
        for i in range(5):
            yield f"chunk_{i}".encode()

    chunks = []
    async for chunk in await iter_with_inactivity_timeout(normal_iterator(), timeout_seconds=1.0):
        chunks.append(chunk)

    assert len(chunks) == 5
    assert chunks[0] == b"chunk_0"


async def test_iter_with_inactivity_timeout_triggers():
    """Test that timeout triggers when no data arrives within threshold."""

    async def slow_iterator():
        """Iterator that stalls after first chunk."""
        yield b"chunk_0"
        # Create a future that never completes (simulates infinite stall)
        await anyio.Event().wait()
        yield b"chunk_1"

    chunks = []
    # Use shorter timeout (0.05s) for faster tests - still validates timeout behavior
    with pytest.raises(InactivityTimeoutError, match="No data received within 0.05 seconds"):
        async for chunk in await iter_with_inactivity_timeout(slow_iterator(), timeout_seconds=0.05):
            chunks.append(chunk)

    # Should have received the first chunk before timeout
    assert len(chunks) == 1
    assert chunks[0] == b"chunk_0"


async def test_iter_with_inactivity_timeout_disabled():
    """Test that None timeout disables the timeout functionality."""

    async def iterator_with_checkpoint():
        """Iterator that would timeout if timeout was enabled."""
        yield b"chunk_0"
        # Checkpoint allows other tasks to run without actual delay
        await anyio.sleep(0)
        yield b"chunk_1"
        await anyio.sleep(0)
        yield b"chunk_2"

    chunks = []
    # With None timeout, should complete successfully
    async for chunk in await iter_with_inactivity_timeout(iterator_with_checkpoint(), timeout_seconds=None):
        chunks.append(chunk)

    assert len(chunks) == 3


async def test_iter_with_inactivity_timeout_empty_iterator():
    """Test timeout with empty iterator completes immediately."""

    async def empty_iterator():
        """Iterator that yields nothing."""
        return
        yield  # Make it a generator

    chunks = []
    async for chunk in await iter_with_inactivity_timeout(empty_iterator(), timeout_seconds=1.0):
        chunks.append(chunk)

    assert len(chunks) == 0


async def test_iter_with_inactivity_timeout_deadline_resets():
    """Test that the inactivity timeout is per gap, not for the whole transfer.

    Each successful chunk read starts a fresh wait window, so transfers where
    total time exceeds the timeout but no single gap does still succeed.
    """
    timeout = 0.1  # 100ms timeout

    async def slow_but_steady_iterator():
        """Iterator with delays shorter than timeout between chunks."""
        for i in range(3):
            if i > 0:
                await anyio.sleep(timeout * 0.7)  # 70% of timeout
            yield f"chunk_{i}".encode()

    # Total time (~140ms) > timeout (100ms), but each gap (70ms) < timeout
    # This should succeed because deadline resets after each chunk
    chunks = []
    async for chunk in await iter_with_inactivity_timeout(slow_but_steady_iterator(), timeout_seconds=timeout):
        chunks.append(chunk)

    assert len(chunks) == 3
    assert chunks == [b"chunk_0", b"chunk_1", b"chunk_2"]


async def test_streaming_file_upload_with_custom_timeout():
    """Test streaming_file_upload respects custom timeout parameter."""
    content = b"x" * 1024

    async def stalling_streaming_body():
        """Stream that sends data then stalls indefinitely."""
        yield content[:512]
        # Stall indefinitely (simulates connection hanging)
        await anyio.Event().wait()
        yield content[512:]

    mock_request = Mock(spec=Request)
    mock_request.headers = {"content-type": "application/octet-stream"}
    mock_request.stream = lambda: stalling_streaming_body()
    mock_request.url = Mock()
    mock_request.url.path = "/test/timeout"

    chunks = []
    # Use shorter timeout (0.05s) for faster tests - still validates timeout behavior
    with pytest.raises((InactivityTimeoutError, ExceptionGroup)) as exc_info:
        async with streaming_file_upload(
            mock_request, OctetStreamChunkProcessor(), inactivity_timeout_seconds=0.05
        ) as upload:
            async for chunk in upload:
                chunks.append(chunk)

    # Verify we got the first chunk before timeout
    assert len(chunks) >= 1

    # Verify InactivityTimeoutError is present
    if isinstance(exc_info.value, ExceptionGroup):
        assert any(isinstance(e, InactivityTimeoutError) for e in exc_info.value.exceptions)


async def test_streaming_file_upload_timeout_disabled():
    """Test streaming_file_upload with timeout disabled (None)."""
    content = b"x" * 1024

    async def streaming_body_with_checkpoints():
        """Stream with checkpoints between chunks."""
        yield content[:256]
        await anyio.sleep(0)  # Checkpoint, no actual delay
        yield content[256:512]
        await anyio.sleep(0)  # Checkpoint, no actual delay
        yield content[512:]

    mock_request = Mock(spec=Request)
    mock_request.headers = {"content-type": "application/octet-stream"}
    mock_request.stream = lambda: streaming_body_with_checkpoints()
    mock_request.url = Mock()
    mock_request.url.path = "/test/no-timeout"

    chunks = []
    # With None timeout, should complete successfully
    async with streaming_file_upload(
        mock_request, OctetStreamChunkProcessor(), inactivity_timeout_seconds=None
    ) as upload:
        async for chunk in upload:
            chunks.append(chunk)

    result = b"".join(chunks)
    assert result == content


# Tests for download_url and download_url_streaming


async def test_download_url_success():
    """Test successful JSON download from URL."""
    mock_response = AsyncMock()
    mock_response.raise_for_status = Mock()
    mock_response.json = AsyncMock(return_value={"status": "ok", "data": "test"})

    mock_session = create_mock_aiohttp_session(mock_response)

    with patch("nhx.core.files.app.streaming.aiohttp.ClientSession", return_value=mock_session):
        result = await download_url("https://example.com/api/data")

        assert result == {"status": "ok", "data": "test"}
        mock_response.raise_for_status.assert_called_once()


async def test_download_url_with_headers():
    """Test download_url with custom headers."""
    mock_response = AsyncMock()
    mock_response.raise_for_status = Mock()
    mock_response.json = AsyncMock(return_value={"authenticated": True})

    mock_session = create_mock_aiohttp_session(mock_response)

    with patch("nhx.core.files.app.streaming.aiohttp.ClientSession", return_value=mock_session):
        result = await download_url("https://example.com/api/data", headers={"Authorization": "Bearer token"})

        assert result == {"authenticated": True}
        # Verify headers were passed
        mock_session.get.assert_called_once()
        call_args = mock_session.get.call_args
        assert call_args[1]["headers"] == {"Authorization": "Bearer token"}


async def test_download_url_streaming_success():
    """Test successful streaming download with injected session."""

    async def mock_chunks(chunk_size):
        yield b"chunk1"
        yield b"chunk2"
        yield b"chunk3"

    mock_response = MagicMock()
    mock_response.raise_for_status = Mock()
    mock_response.closed = False
    mock_response.content.iter_chunked = mock_chunks

    mock_session = create_mock_aiohttp_session(mock_response)

    chunks = []
    async for chunk in download_url_streaming("https://example.com/file.txt", session=mock_session):
        chunks.append(chunk)

    assert chunks == [b"chunk1", b"chunk2", b"chunk3"]
    mock_response.raise_for_status.assert_called_once()


async def test_download_url_streaming_with_byte_range():
    """Test streaming download with byte range."""

    async def mock_chunks(chunk_size):
        yield b"partial"

    mock_response = MagicMock()
    mock_response.raise_for_status = Mock()
    mock_response.closed = False
    mock_response.content.iter_chunked = mock_chunks

    mock_session = create_mock_aiohttp_session(mock_response)

    byte_range = ByteRange(start=0, end=100)
    chunks = []
    async for chunk in download_url_streaming(
        "https://example.com/file.txt", byte_range=byte_range, session=mock_session
    ):
        chunks.append(chunk)

    # Verify Range header was set
    mock_session.get.assert_called_once()
    call_args = mock_session.get.call_args
    assert call_args[1]["headers"]["Range"] == "bytes=0-100"


async def test_download_url_streaming_cleanup_on_error():
    """Test that response is properly cleaned up on error via context manager."""

    async def mock_chunks_with_error(chunk_size):
        yield b"chunk1"
        raise Exception("Download error")

    mock_response = MagicMock()
    mock_response.raise_for_status = Mock()
    mock_response.content.iter_chunked = mock_chunks_with_error

    mock_session = create_mock_aiohttp_session(mock_response)

    with pytest.raises(Exception, match="Download error"):
        async for _ in download_url_streaming("https://example.com/file.txt", session=mock_session):
            pass

    # Verify response context manager exited (cleanup)
    mock_response.__aexit__.assert_called_once()


async def test_download_url_streaming_does_not_close_injected_session():
    """Test that an injected session is not closed by the function."""

    async def mock_chunks(chunk_size):
        yield b"data"

    mock_response = MagicMock()
    mock_response.raise_for_status = Mock()
    mock_response.content.iter_chunked = mock_chunks

    mock_session = create_mock_aiohttp_session(mock_response)
    mock_session.close = AsyncMock()

    async for _ in download_url_streaming("https://example.com/file.txt", session=mock_session):
        pass

    # Injected session should NOT be closed (nullcontext doesn't call __aexit__)
    mock_session.__aexit__.assert_not_called()


async def test_download_url_streaming_applies_no_total_timeout_per_request():
    """Test that requests have no total timeout (for large downloads)."""

    async def mock_chunks(chunk_size):
        yield b"data"

    mock_response = MagicMock()
    mock_response.raise_for_status = Mock()
    mock_response.content.iter_chunked = mock_chunks

    mock_session = create_mock_aiohttp_session(mock_response)

    # Test with injected session - timeout should still be applied per-request
    async for _ in download_url_streaming("https://example.com/file.txt", session=mock_session):
        pass

    # Verify timeout was passed to session.get()
    mock_session.get.assert_called_once()
    call_kwargs = mock_session.get.call_args[1]
    assert call_kwargs["timeout"].total is None


# Tests for URL encoding preservation in download functions


async def test_download_url_preserves_encoded_url():
    """Test that download_url doesn't re-encode already-encoded URL parameters.

    Signed URLs (e.g., from NGC) contain percent-encoded query parameters
    with cryptographic signatures. Re-encoding these invalidates the signature,
    causing 403 Forbidden from the storage backend.
    """
    from yarl import URL

    mock_response = AsyncMock()
    mock_response.raise_for_status = Mock()
    mock_response.json = AsyncMock(return_value={"urls": ["https://signed.example.com"]})

    mock_session = create_mock_aiohttp_session(mock_response)

    # URL with already-encoded params (like NGC signed URLs)
    encoded_url = (
        "https://api.ngc.nvidia.com/v2/models/org/nvidia/team/nemo/test/1.0/files?path=model.yaml&key=a%2Bb%3Dc"
    )

    with patch("nhx.core.files.app.streaming.aiohttp.ClientSession", return_value=mock_session):
        await download_url(encoded_url, headers={"Authorization": "Bearer token"})

    # Verify the URL was passed with encoded=True (yarl URL object)
    call_args = mock_session.get.call_args
    url_arg = call_args[0][0] if call_args[0] else call_args[1].get("url")
    assert isinstance(url_arg, URL)
    # The encoded characters should be preserved, not double-encoded
    assert "%2B" in str(url_arg)
    assert "%3D" in str(url_arg)


async def test_download_url_streaming_preserves_encoded_url():
    """Test that download_url_streaming doesn't re-encode signed URLs.

    This is the critical fix for NGC model downloads: aiohttp's default URL
    handling re-encodes percent-encoded query params in signed URLs, which
    invalidates the cryptographic signature and causes 403 from xfiles.ngc.nvidia.com.
    """
    from yarl import URL

    async def mock_chunks(chunk_size):
        yield b"data"

    mock_response = MagicMock()
    mock_response.raise_for_status = Mock()
    mock_response.content.iter_chunked = mock_chunks

    mock_session = create_mock_aiohttp_session(mock_response)

    # Signed URL with percent-encoded params (like NGC xfiles URLs)
    signed_url = "https://xfiles.ngc.nvidia.com/org/nvidia/team/nemo/models/test/versions/1.0/files/model.yaml?ssec-key=a%2Bb&Signature=x~y~z"

    async for _ in download_url_streaming(signed_url, session=mock_session):
        pass

    # Verify the URL was passed with encoded=True (yarl URL object)
    call_args = mock_session.get.call_args
    url_arg = call_args[0][0] if call_args[0] else call_args[1].get("url")
    assert isinstance(url_arg, URL)
    # The encoded characters and tilde should be preserved exactly
    assert "%2B" in str(url_arg)
    assert "~" in str(url_arg)


# Tests for tee_stream


async def async_iter_from_list(items: list[bytes]):
    """Helper to create an async iterator from a list of bytes."""
    for item in items:
        yield item


async def test_tee_stream_basic():
    """Test tee_stream distributes data correctly to both streams."""

    async def consume_both(stream1, stream2):
        results1, results2 = [], []
        async with anyio.create_task_group() as tg:

            async def consume1():
                async for chunk in stream1:
                    results1.append(chunk)

            async def consume2():
                async for chunk in stream2:
                    results2.append(chunk)

            tg.start_soon(consume1)
            tg.start_soon(consume2)
        return results1, results2

    # Test with normal data - both streams receive same data in order
    chunks = [f"chunk_{i:03d}".encode() for i in range(20)]
    async with tee_stream(async_iter_from_list(chunks)) as (s1, s2):
        r1, r2 = await consume_both(s1, s2)
    assert r1 == r2 == chunks

    # Test with empty input
    async def empty_iter():
        return
        yield

    async with tee_stream(empty_iter()) as (s1, s2):
        r1, r2 = await consume_both(s1, s2)
    assert r1 == r2 == []

    # Test with large data (100 chunks of 1KB)
    large_chunks = [b"x" * 1000 for _ in range(100)]
    async with tee_stream(async_iter_from_list(large_chunks)) as (s1, s2):
        r1, r2 = await consume_both(s1, s2)
    assert b"".join(r1) == b"".join(r2) == b"".join(large_chunks)


async def test_tee_stream_integration_with_streaming_upload():
    """Test tee_stream with streaming_file_upload (typical cache-while-streaming use case)."""
    content = b"test file content " * 100
    request = create_streaming_request(content)

    disk_chunks, response_chunks = [], []

    async with streaming_file_upload(request, OctetStreamChunkProcessor()) as upload:
        async with tee_stream(upload) as (disk_stream, response_stream):
            async with anyio.create_task_group() as tg:

                async def write_to_disk():
                    async for chunk in disk_stream:
                        disk_chunks.append(chunk)

                tg.start_soon(write_to_disk)

                async for chunk in response_stream:
                    response_chunks.append(chunk)

    assert b"".join(disk_chunks) == b"".join(response_chunks) == content


async def test_tee_stream_slow_consumer_timeout():
    """Test that InactivityTimeoutError is raised when a consumer stops reading."""
    chunks = [b"chunk"] * 20  # More chunks than queue size

    with pytest.raises(InactivityTimeoutError, match="Consumer not reading fast enough"):
        async with tee_stream(
            async_iter_from_list(chunks),
            queue_size=2,
            timeout_seconds=0.1,
        ) as (s1, _s2):
            # Only consume s1, ignore s2 - should timeout
            async for _ in s1:
                pass


# Tests for preflight=True behavior


async def test_iter_with_inactivity_timeout_preflight_yields_all_chunks():
    """Test that preflight=True yields first chunk followed by remaining chunks."""

    async def mock_chunks():
        yield b"first"
        yield b"second"
        yield b"third"

    # preflight=True reads the first chunk during the await call
    stream = await iter_with_inactivity_timeout(mock_chunks(), timeout_seconds=1.0, preflight=True)

    # Iterating yields all chunks (first chunk was read during preflight but is still yielded)
    chunks = [chunk async for chunk in stream]
    assert chunks == [b"first", b"second", b"third"]


async def test_iter_with_inactivity_timeout_preflight_empty_iterator():
    """Test that preflight=True handles empty iterator."""

    async def empty_iter():
        return
        yield  # pragma: no cover

    # preflight=True reads the first chunk (nothing) during await
    stream = await iter_with_inactivity_timeout(empty_iter(), timeout_seconds=1.0, preflight=True)

    # Iterator should be empty
    chunks = [chunk async for chunk in stream]
    assert chunks == []


async def test_iter_with_inactivity_timeout_preflight_propagates_exception():
    """Test that exceptions during preflight read propagate to caller."""

    async def error_on_first():
        raise ConnectionError("Network failure")
        yield  # pragma: no cover

    # Exception should be raised during the await (during preflight)
    with pytest.raises(ConnectionError, match="Network failure"):
        await iter_with_inactivity_timeout(error_on_first(), timeout_seconds=1.0, preflight=True)


async def test_iter_with_inactivity_timeout_preflight_timeout():
    """Test that timeout works during preflight."""

    async def slow_first():
        await anyio.Event().wait()  # Never completes
        yield b"never"

    # Timeout should be raised during the await (during preflight)
    with pytest.raises(InactivityTimeoutError, match="No data received within 0.05 seconds"):
        await iter_with_inactivity_timeout(slow_first(), timeout_seconds=0.05, preflight=True)
