# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
from types import SimpleNamespace

import anyio
import pytest
from aiohttp import web
from nhx.common.api.common import SecretRef
from nhx.common.files.storage_config import S3StorageConfig
from nhx.core.files.app.backends.s3 import S3AccessError, S3ConfigError, S3StorageImpl, S3UnavailableError


@pytest.fixture
async def managed_upload(monkeypatch):
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
    for start in range(0, len(data), 3 * 1024**2):
        yield b""
        yield data[start : start + 3 * 1024**2]


@pytest.mark.parametrize("size", [0, 7, 17 * 1024**2])
async def test_managed_transfer_retains_bytes(managed_upload, size):
    impl, state = managed_upload
    data = (b"test-data" * (size // 9 + 1))[:size]
    result = await impl.upload("file", chunks(data), size)
    assert state.saved == data and (result.path, result.size) == ("file", size)
    assert all("Credential=explicit/" in auth for auth in state.auth)


@pytest.mark.parametrize("failures", [1, 2])
async def test_managed_retries_and_failed_part_abort(managed_upload, monkeypatch, failures):
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
    [(404, "NoSuchBucket", S3ConfigError), (403, "AccessDenied", S3AccessError), (400, "ExpiredToken", S3AccessError)],
)
async def test_storage_rejection_preserves_error_kind(managed_upload, status, code, error):
    impl, state = managed_upload
    state.rejection = status, code
    with pytest.raises(error):
        await impl.upload("file", chunks(b"abc"), 3)
    assert state.saved is None


@pytest.mark.parametrize("declared", [2, 5])
async def test_mismatched_length_fails_before_object_is_saved(managed_upload, declared):
    impl, state = managed_upload
    with pytest.raises(ValueError, match="Content-Length"):
        await impl.upload("file", chunks(b"abc"), declared)
    assert state.saved is None


async def test_source_error_aborts_partial_upload(managed_upload):
    impl, state = managed_upload

    async def failed_source():
        yield b"x" * (8 * 1024**2)
        raise OSError("source failed")

    with pytest.raises(OSError, match="source failed"):
        await impl.upload("file", failed_source(), 9 * 1024**2)
    assert state.aborted.is_set() and state.saved is None


@pytest.mark.parametrize("raw_cancel", [False, True])
async def test_cancelled_source_aborts(managed_upload, raw_cancel):
    impl, state = managed_upload
    waiting, release = anyio.Event(), anyio.Event()

    async def source():
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
