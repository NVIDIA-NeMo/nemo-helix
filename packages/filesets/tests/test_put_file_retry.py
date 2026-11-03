# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""put_file resends a local file only for failures the client could not retry itself.

The client retries a connection that never sent the body, and put_file resends the
whole file after a retryable failure once the body has gone out. These tests cover
the failures put_file must hand straight back, so the two retry loops never stack.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable
from pathlib import Path

import httpx
import pytest
from filesets import UPLOAD_RETRY_POLICY, UploadRetriesExhaustedError
from filesets.filesystem.filesystem import AsyncFilesetFileSystem, FilesetFileSystem
from nemo_helix_plugin.client.errors import NemoHTTPError, NemoTransportError, PermissionDeniedError
from nemo_helix_plugin.files.client import AsyncFilesClient, FilesClient

BASE = "http://test:8000"
REMOTE = "default/set/upload.bin"
# UPLOAD_RETRY_POLICY's status rules and attempt count, without the backoff waits.
POLICY = dataclasses.replace(UPLOAD_RETRY_POLICY, backoff_base=0.0)
ATTEMPTS = POLICY.max_retries + 1


def _connect_error(request: httpx.Request) -> httpx.Response:
    raise httpx.ConnectError("connection refused", request=request)


def _status(code: int) -> Callable[[httpx.Request], httpx.Response]:
    return lambda request: httpx.Response(code, json={"detail": "nope"}, request=request)


# (handler, expected error, expected upload requests)
_CASES = [
    # The client already retried a connection that never read the body; resending
    # would multiply its attempts by put_file's.
    pytest.param(_connect_error, NemoTransportError, ATTEMPTS, id="unsent-transport-error"),
    # Not retryable under the policy: fail on the first upload.
    pytest.param(_status(403), PermissionDeniedError, 1, id="non-retryable-status"),
    # A 5xx the policy would retry, but the server is out of space.
    pytest.param(_status(507), NemoHTTPError, 1, id="insufficient-storage"),
    # Control: a retryable status after the body went out is resent by put_file.
    pytest.param(_status(503), UploadRetriesExhaustedError, ATTEMPTS, id="retryable-status"),
]


@pytest.fixture
def local_file(tmp_path: Path) -> Path:
    path = tmp_path / "upload.bin"
    path.write_bytes(b"x" * 4096)
    return path


class _Recorder:
    def __init__(self, handler: Callable[[httpx.Request], httpx.Response]) -> None:
        self.handler = handler
        self.uploads = 0

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.uploads += 1
        return self.handler(request)


@pytest.mark.parametrize("handler,error,uploads", _CASES)
def test_sync_put_file_does_not_stack_retries(
    local_file: Path,
    handler: Callable[[httpx.Request], httpx.Response],
    error: type[Exception],
    uploads: int,
) -> None:
    recorder = _Recorder(handler)
    http = httpx.Client(transport=httpx.MockTransport(recorder), base_url=BASE)
    client = FilesClient(base_url=BASE, workspace="default", http_client=http, retry=POLICY)
    fs = FilesetFileSystem(client=client, blocksize=1024, skip_instance_cache=True)

    with pytest.raises(error):
        fs.put_file(str(local_file), REMOTE)

    assert recorder.uploads == uploads


@pytest.mark.parametrize("handler,error,uploads", _CASES)
async def test_async_put_file_does_not_stack_retries(
    local_file: Path,
    handler: Callable[[httpx.Request], httpx.Response],
    error: type[Exception],
    uploads: int,
) -> None:
    recorder = _Recorder(handler)
    async with httpx.AsyncClient(transport=httpx.MockTransport(recorder), base_url=BASE) as http:
        client = AsyncFilesClient(base_url=BASE, workspace="default", http_client=http, retry=POLICY)
        fs = AsyncFilesetFileSystem(client=client, blocksize=1024, skip_instance_cache=True)

        with pytest.raises(error):
            await fs._put_file(str(local_file), REMOTE)

    assert recorder.uploads == uploads


_MID_BODY_ERRORS = [
    pytest.param(httpx.WriteTimeout, id="write-timeout"),
    pytest.param(httpx.ReadTimeout, id="read-timeout"),
]


class _MidBodyFailures(httpx.BaseTransport, httpx.AsyncBaseTransport):
    """Reads one chunk of each upload body, then fails, for the first ``failures`` uploads.

    The uploads after that read the whole body and succeed; ``bodies`` keeps what each
    successful upload received.
    """

    def __init__(self, error: type[httpx.TransportError], failures: int) -> None:
        self.error = error
        self.failures = failures
        self.uploads = 0
        self.bodies: list[bytes] = []

    def _fail_or_ok(self, request: httpx.Request, body: bytes | None) -> httpx.Response:
        if body is None:
            raise self.error("connection dropped mid-upload", request=request)
        self.bodies.append(body)
        payload = {"file_ref": "ref", "file_url": "url", "path": "upload.bin", "size": len(body)}
        return httpx.Response(200, json=payload, request=request)

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        self.uploads += 1
        assert isinstance(request.stream, httpx.SyncByteStream)
        chunks = iter(request.stream)
        if self.uploads <= self.failures:
            next(chunks)
            return self._fail_or_ok(request, None)
        return self._fail_or_ok(request, b"".join(chunks))

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        self.uploads += 1
        assert isinstance(request.stream, httpx.AsyncByteStream)
        chunks = aiter(request.stream)
        if self.uploads <= self.failures:
            await anext(chunks)
            return self._fail_or_ok(request, None)
        return self._fail_or_ok(request, b"".join([chunk async for chunk in chunks]))


@pytest.fixture
def payload_file(tmp_path: Path) -> tuple[Path, bytes]:
    payload = bytes(range(256)) * 16
    path = tmp_path / "payload.bin"
    path.write_bytes(payload)
    return path, payload


def _sync_fs(transport: httpx.BaseTransport) -> FilesetFileSystem:
    http = httpx.Client(transport=transport, base_url=BASE)
    client = FilesClient(base_url=BASE, workspace="default", http_client=http, retry=POLICY)
    return FilesetFileSystem(client=client, blocksize=1024, skip_instance_cache=True)


def _async_fs(http: httpx.AsyncClient) -> AsyncFilesetFileSystem:
    client = AsyncFilesClient(base_url=BASE, workspace="default", http_client=http, retry=POLICY)
    return AsyncFilesetFileSystem(client=client, blocksize=1024, skip_instance_cache=True)


def _assert_exhausted_by(raised: UploadRetriesExhaustedError, error: type[httpx.TransportError]) -> None:
    assert raised.attempts == ATTEMPTS
    assert isinstance(raised.__cause__, NemoTransportError)
    assert isinstance(raised.__cause__.error, error)


@pytest.mark.parametrize("error", _MID_BODY_ERRORS)
def test_sync_put_file_resends_whole_file_after_mid_body_failure(
    payload_file: tuple[Path, bytes], error: type[httpx.TransportError]
) -> None:
    path, payload = payload_file
    transport = _MidBodyFailures(error, failures=1)

    _sync_fs(transport).put_file(str(path), REMOTE)

    assert transport.uploads == 2
    assert transport.bodies == [payload]


@pytest.mark.parametrize("error", _MID_BODY_ERRORS)
async def test_async_put_file_resends_whole_file_after_mid_body_failure(
    payload_file: tuple[Path, bytes], error: type[httpx.TransportError]
) -> None:
    path, payload = payload_file
    transport = _MidBodyFailures(error, failures=1)

    async with httpx.AsyncClient(transport=transport, base_url=BASE) as http:
        await _async_fs(http)._put_file(str(path), REMOTE)

    assert transport.uploads == 2
    assert transport.bodies == [payload]


@pytest.mark.parametrize("error", _MID_BODY_ERRORS)
def test_sync_put_file_gives_up_after_repeated_mid_body_failures(
    payload_file: tuple[Path, bytes], error: type[httpx.TransportError]
) -> None:
    path, _ = payload_file
    transport = _MidBodyFailures(error, failures=ATTEMPTS + 1)

    with pytest.raises(UploadRetriesExhaustedError) as raised:
        _sync_fs(transport).put_file(str(path), REMOTE)

    assert transport.uploads == ATTEMPTS
    _assert_exhausted_by(raised.value, error)


@pytest.mark.parametrize("error", _MID_BODY_ERRORS)
async def test_async_put_file_gives_up_after_repeated_mid_body_failures(
    payload_file: tuple[Path, bytes], error: type[httpx.TransportError]
) -> None:
    path, _ = payload_file
    transport = _MidBodyFailures(error, failures=ATTEMPTS + 1)

    async with httpx.AsyncClient(transport=transport, base_url=BASE) as http:
        with pytest.raises(UploadRetriesExhaustedError) as raised:
            await _async_fs(http)._put_file(str(path), REMOTE)

    assert transport.uploads == ATTEMPTS
    _assert_exhausted_by(raised.value, error)


def test_sync_put_file_raises_original_connect_error(local_file: Path) -> None:
    recorder = _Recorder(_connect_error)
    fs = _sync_fs(httpx.MockTransport(recorder))

    with pytest.raises(NemoTransportError) as raised:
        fs.put_file(str(local_file), REMOTE)

    # Only the client's own attempts; put_file adds none and does not wrap the error.
    assert recorder.uploads == ATTEMPTS
    assert type(raised.value) is NemoTransportError
    assert isinstance(raised.value.error, httpx.ConnectError)


async def test_async_put_file_raises_original_connect_error(local_file: Path) -> None:
    recorder = _Recorder(_connect_error)
    async with httpx.AsyncClient(transport=httpx.MockTransport(recorder), base_url=BASE) as http:
        with pytest.raises(NemoTransportError) as raised:
            await _async_fs(http)._put_file(str(local_file), REMOTE)

    assert recorder.uploads == ATTEMPTS
    assert type(raised.value) is NemoTransportError
    assert isinstance(raised.value.error, httpx.ConnectError)
