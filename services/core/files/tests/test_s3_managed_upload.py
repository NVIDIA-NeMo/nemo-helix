# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Exercise managed S3 transfers against a loopback endpoint using the real SDK."""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import anyio
import pytest
from aiohttp import web
from fastapi import HTTPException, Request
from nhx.common.api.common import SecretRef
from nhx.common.files.storage_config import S3StorageConfig
from nhx.core.files.api.endpoint_helpers import UPSTREAM_STORAGE_ACCESS_DETAIL
from nhx.core.files.api.v2.filesets import endpoints
from nhx.core.files.app.backends.s3 import S3AccessError, S3ConfigError, S3StorageImpl, S3UnavailableError


@pytest.fixture
async def managed_upload(monkeypatch):
    """Provide a local S3 endpoint with controllable retries, errors, and cancellation."""
    monkeypatch.setenv("AWS_EC2_METADATA_DISABLED", "true")
    monkeypatch.setenv("AWS_MAX_ATTEMPTS", "2")
    monkeypatch.setenv("AWS_RETRY_MODE", "standard")
    state = SimpleNamespace(
        parts={},
        saved=None,
        attempts=0,
        failures=0,
        rejection=None,
        auth=[],
        aborted=anyio.Event(),
        finished=anyio.Event(),
        block_part=False,
        entered=anyio.Event(),
        release=anyio.Event(),
    )

    async def handler(request):
        """Record PUT/multipart requests and simulate the selected storage failures."""
        assert request.path == "/test/prefix/file"
        if state.rejection:
            status, code = state.rejection
            return web.Response(status=status, text=f"<Error><Code>{code}</Code></Error>")
        state.auth.append(request.headers.get("Authorization", ""))
        if request.method == "DELETE":
            state.aborted.set()
            state.finished.set()
            return web.Response(status=204)
        if request.method == "PUT":
            data = await request.read()
            if "partNumber" not in request.query:
                state.saved = data
                state.finished.set()
                return web.Response()
            state.attempts += 1
            if state.block_part:
                state.entered.set()
                await state.release.wait()
            if state.attempts <= state.failures:
                return web.Response(status=503, text="<Error><Code>SlowDown</Code></Error>")
            state.parts[int(request.query["partNumber"])] = data
            return web.Response(headers={"ETag": '"part"'})
        if "uploads" in request.query:
            return web.Response(
                text="<InitiateMultipartUploadResult><UploadId>u</UploadId></InitiateMultipartUploadResult>"
            )
        await request.read()
        state.saved = b"".join(state.parts[number] for number in sorted(state.parts))
        state.finished.set()
        return web.Response(
            text='<CompleteMultipartUploadResult><ETag>"complete"</ETag></CompleteMultipartUploadResult>'
        )

    app = web.Application(client_max_size=32 * 1024**2)
    app.router.add_route("*", "/{path:.*}", handler)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    impl = S3StorageImpl(
        S3StorageConfig(
            bucket="test",
            prefix="prefix",
            region="us-east-1",
            endpoint_url=f"http://127.0.0.1:{runner.addresses[0][1]}",
            use_sdk_auth=False,
            access_key_id_secret=SecretRef(root="key"),
            secret_access_key_secret=SecretRef(root="secret"),
        ),
        {"access_key_id": "explicit", "secret_access_key": "test"},
    )
    try:
        yield impl, state
    finally:
        await runner.cleanup()


async def chunks(data):
    """Yield request-sized fragments interleaved with empty chunks."""
    for start in range(0, len(data), 3 * 1024**2):
        yield b""
        yield data[start : start + 3 * 1024**2]


@pytest.mark.parametrize("size", [0, 7, 17 * 1024**2])
async def test_managed_transfer_retains_bytes(managed_upload, size):
    """Empty, small, and multipart uploads preserve bytes and explicit credentials."""
    impl, state = managed_upload
    data = (b"test-data" * (size // 9 + 1))[:size]
    result = await impl.upload("file", chunks(data), size)
    assert state.saved == data and (result.path, result.size) == ("file", size)
    assert all("Credential=explicit/" in auth for auth in state.auth)


@pytest.mark.parametrize("failures", [1, 2])
async def test_managed_retries_and_failed_part_abort(managed_upload, monkeypatch, failures):
    """Retry transient part failures and abort after retry exhaustion using SDK credentials."""
    impl, state = managed_upload
    impl.config.use_sdk_auth = True
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "ambient")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "test")
    state.failures = failures
    data = b"x" * (17 * 1024**2)
    with anyio.fail_after(5):
        if failures == 1:
            await impl.upload("file", chunks(data), len(data))
            assert state.saved == data and state.attempts == len(state.parts) + 1
        else:
            with pytest.raises(S3UnavailableError):
                await impl.upload("file", chunks(data), len(data))
            assert state.aborted.is_set() and state.saved is None and state.attempts == 2
    assert all("Credential=ambient/" in auth for auth in state.auth)


@pytest.mark.parametrize(
    "status,code,error",
    [
        (404, "NoSuchBucket", S3ConfigError),
        (403, "AccessDenied", S3AccessError),
        (400, "ExpiredToken", S3AccessError),
        (403, "InvalidSecurity", S3AccessError),
    ],
)
async def test_storage_rejection_preserves_error_kind(managed_upload, status, code, error):
    """Retain storage error classifications when the backend rejects an upload."""
    impl, state = managed_upload
    state.rejection = status, code
    with pytest.raises(error):
        await impl.upload("file", chunks(b"abc"), 3)
    assert state.saved is None


@pytest.mark.parametrize("declared", [2, 5])
async def test_mismatched_length_fails_before_object_is_saved(managed_upload, declared):
    """Reject both truncated and oversized bodies without publishing an object."""
    impl, state = managed_upload
    with pytest.raises(ValueError, match="Content-Length"):
        await impl.upload("file", chunks(b"abc"), declared)
    assert state.saved is None


async def test_source_error_aborts_partial_upload(managed_upload):
    """A failing request stream aborts an already-started multipart transfer."""
    impl, state = managed_upload

    async def failed_source():
        """Fail after supplying enough data to initiate a multipart upload."""
        yield b"x" * (8 * 1024**2)
        raise OSError("source failed")

    with pytest.raises(OSError, match="source failed"):
        await impl.upload("file", failed_source(), 9 * 1024**2)
    assert state.aborted.is_set() and state.saved is None


@pytest.mark.parametrize("raw_cancel", [False, True])
async def test_cancelled_source_aborts(managed_upload, raw_cancel):
    """Both task and task-group cancellation abort a transfer waiting on its source."""
    impl, state = managed_upload
    waiting, release = anyio.Event(), anyio.Event()

    async def source():
        """Pause the request stream after multipart transfer initiation."""
        yield b"x" * (8 * 1024**2)
        waiting.set()
        await release.wait()
        yield b"x"

    with anyio.fail_after(3):
        if raw_cancel:
            task = asyncio.create_task(impl.upload("file", source(), 8 * 1024**2 + 1))
            await waiting.wait()
            task.cancel()
            try:
                with pytest.raises(asyncio.CancelledError):
                    await task
                with anyio.fail_after(1):
                    await state.aborted.wait()
            finally:
                release.set()
                with anyio.move_on_after(1):
                    await state.finished.wait()
        else:
            async with anyio.create_task_group() as group:
                group.start_soon(impl.upload, "file", source(), 8 * 1024**2 + 1)
                await waiting.wait()
                group.cancel_scope.cancel()
    assert state.aborted.is_set() and state.saved is None


async def test_network_cancellation_waits_for_request_then_aborts(managed_upload):
    """Cancellation during a part request aborts once synchronous network I/O returns."""
    impl, state = managed_upload
    state.block_part = True
    data = b"x" * (17 * 1024**2)
    with anyio.fail_after(5):
        task = asyncio.create_task(impl.upload("file", chunks(data), len(data)))
        await state.entered.wait()
        task.cancel()
        state.release.set()
        with pytest.raises(asyncio.CancelledError):
            await task
    assert state.aborted.is_set() and state.saved is None


@pytest.mark.parametrize("multipart", [False, True])
@pytest.mark.parametrize("size", [0, 7, 17 * 1024**2])
async def test_endpoint_upload_lengths(managed_upload, monkeypatch, multipart, size):
    """Raw and multipart requests save only file bytes and report their exact size."""
    impl, state = managed_upload
    data = b"x" * size
    body = data
    content_type = b"application/octet-stream"
    if multipart:
        content_type = b"multipart/form-data; boundary=test-boundary"
        body = (
            b'--test-boundary\r\nContent-Disposition: form-data; name="file"; filename="file"\r\n'
            b"Content-Type: application/octet-stream\r\n\r\n" + data + b"\r\n--test-boundary--\r\n"
        )
    request = upload_request(body, len(body), content_type)
    configure_endpoint(monkeypatch, impl)
    result = await endpoints.upload_file("ws", "files", "file", request, MagicMock(), MagicMock(), MagicMock())
    assert result.size == size
    assert state.saved == data


def configure_endpoint(monkeypatch, impl):
    """Resolve a fileset to the real loopback S3 backend."""
    fileset = SimpleNamespace(name="files", storage=impl.config, custom_fields={})
    monkeypatch.setattr(endpoints, "get_fileset", AsyncMock(return_value=fileset))
    monkeypatch.setattr(endpoints, "resolve_storage_secrets_for_user", AsyncMock(return_value={}))
    monkeypatch.setattr(endpoints, "storage_impl_factory", lambda *args: impl)


def upload_request(body, declared, content_type=b"application/octet-stream"):
    """Build a request whose ASGI body can disagree with its declared size."""
    fragments = iter([body[i : i + 65536] for i in range(0, len(body), 65536)])

    async def receive():
        chunk = next(fragments, None)
        return {"type": "http.request", "body": chunk or b"", "more_body": chunk is not None}

    return Request(
        {
            "type": "http",
            "path": "/upload",
            "headers": [(b"content-type", content_type), (b"content-length", str(declared).encode())],
        },
        receive,
    )


@pytest.mark.parametrize("multipart", [False, True])
@pytest.mark.parametrize("delta", [-1, 1])
async def test_endpoint_rejects_request_length_mismatch(managed_upload, monkeypatch, multipart, delta):
    """Validate framing bytes separately and retain raw-body length checks."""
    impl, state = managed_upload
    configure_endpoint(monkeypatch, impl)
    body = b"abc"
    content_type = b"application/octet-stream"
    if multipart:
        content_type = b"multipart/form-data; boundary=b"
        body = b'--b\r\nContent-Disposition: form-data; name="file"; filename="f"\r\n\r\nabc\r\n--b--\r\n'
    with pytest.raises(HTTPException) as caught:
        await endpoints.upload_file(
            "ws",
            "files",
            "file",
            upload_request(body, len(body) + delta, content_type),
            MagicMock(),
            MagicMock(),
            MagicMock(),
        )
    assert caught.value.status_code == 400
    assert state.saved is None


async def test_endpoint_invalid_security_is_generic_403(managed_upload, monkeypatch):
    """Provider InvalidSecurity responses reach the redacted access-error handler."""
    impl, state = managed_upload
    configure_endpoint(monkeypatch, impl)
    state.rejection = 403, "InvalidSecurity"
    with pytest.raises(HTTPException) as caught:
        await endpoints.upload_file(
            "ws", "files", "file", upload_request(b"abc", 3), MagicMock(), MagicMock(), MagicMock()
        )
    assert caught.value.status_code == 403
    assert caught.value.detail == UPSTREAM_STORAGE_ACCESS_DETAIL
