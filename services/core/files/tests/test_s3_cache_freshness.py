# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""S3 freshness, metadata TTL and authorization regressions using a fake S3 transport."""

import hashlib
import json
from collections import OrderedDict
from contextlib import asynccontextmanager
from copy import copy
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import aiohttp
import anyio
import httpx
import pytest
from aiobotocore.credentials import AioCredentials
from botocore.exceptions import ClientError
from fastapi import BackgroundTasks, FastAPI, HTTPException
from nemo_helix_plugin.files.types import CacheStatus
from nhx.common.files.storage_config import S3StorageConfig
from nhx.core.files.api.endpoint_helpers import (
    UPSTREAM_STORAGE_ACCESS_DETAIL,
    UPSTREAM_STORAGE_UNAVAILABLE_DETAIL,
    CacheContext,
    download_with_cache,
    get_cache_status_for_files,
    resolve_download_storage,
)
from nhx.core.files.api.v2.filesets import endpoints
from nhx.core.files.api.v2.hf import endpoints as hf_endpoints
from nhx.core.files.app.backends import s3 as s3_backend
from nhx.core.files.app.backends.base import FileInfo
from nhx.core.files.app.backends.local import LocalStorageConfig, LocalStorageImpl
from nhx.core.files.app.backends.s3 import S3BackendError, S3StorageImpl, S3UnavailableError
from nhx.core.files.app.cache import cache_file_directly, warm_fileset_cache
from nhx.core.files.config import FilesConfig
from nhx.core.files.exceptions import NotFoundError, StorageObjectChangedError


async def chunks(data):
    yield data


@pytest.fixture
def env(monkeypatch, tmp_path):
    """Start with object A, real local storage, and isolated metadata/clock state."""
    monkeypatch.setattr(s3_backend, "_metadata_cache", OrderedDict())
    monkeypatch.setattr(s3_backend, "_metadata_generation", 0)
    config = FilesConfig(s3_metadata_cache_ttl_seconds=0)
    monkeypatch.setattr("nhx.core.files.config.files_config", lambda: config)
    now = [100.0]
    monkeypatch.setattr(s3_backend, "monotonic", lambda: now[0])
    storage = S3StorageImpl(S3StorageConfig(bucket="bucket", prefix="prefix", use_sdk_auth=True), {})
    credentials = AioCredentials("test", "upstream-secret")
    monkeypatch.setattr(storage._session._session, "get_credentials", AsyncMock(return_value=credentials))
    objects, versions, downloads, held_locks = {}, {}, [], set()
    permissions = {"versions": True, "content": True}

    def put(data, metadata=None):
        """Write directly to the bucket, retaining any historical version."""
        metadata = metadata if metadata is not None else {"ETag": f'"{hashlib.md5(data).hexdigest()}"'}
        objects["prefix/data.jsonl"] = (data, metadata)
        if metadata.get("VersionId"):
            versions[metadata["VersionId"]] = (data, metadata)

    async def head_object(*, Bucket, Key):
        if Key not in objects:
            raise ClientError({"Error": {"Code": "404"}}, "HeadObject")
        data, metadata = objects[Key]
        return dict(metadata, ContentLength=len(data))

    def upload(source, bucket, key, *, Config):
        """Consume the managed transfer stream on its AnyIO worker."""
        assert bucket == "bucket" and key == "prefix/data.jsonl"
        put(source.read())

    async def download(url, *, session, headers, byte_range, chunk_size):
        """Enforce S3 version permissions and If-Match at the network boundary."""
        params = json.loads(url)
        downloads.append(params)
        if not permissions["content"] or ("VersionId" in params and not permissions["versions"]):
            raise aiohttp.ClientResponseError(MagicMock(), (), status=403)
        obj = versions.get(params["VersionId"]) if "VersionId" in params else objects.get(params["Key"])
        if obj is None:
            raise NotFoundError("Object deleted")
        data, metadata = obj
        if "IfMatch" in params:
            assert headers["If-Match"] == params["IfMatch"]
            if params["IfMatch"] != metadata.get("ETag"):
                raise aiohttp.ClientResponseError(MagicMock(), (), status=412)
        yield data if byte_range is None else data[byte_range.start : byte_range.end + 1]

    client = AsyncMock()
    client.head_object.side_effect = head_object
    client.generate_presigned_url.side_effect = lambda operation, Params, ExpiresIn: json.dumps(Params)
    client.delete_object.side_effect = lambda Bucket, Key: objects.pop(Key, None)
    session_client = MagicMock()
    session_client.return_value.__aenter__.return_value = client
    monkeypatch.setattr(storage._session, "client", session_client)
    upload_client = MagicMock()
    upload_client.upload_fileobj.side_effect = upload
    upload_session = MagicMock()
    upload_session.client.return_value = upload_client
    monkeypatch.setattr(s3_backend.boto3, "Session", lambda: upload_session)
    monkeypatch.setattr(s3_backend, "download_url_streaming", download)
    monkeypatch.setattr(s3_backend, "get_http_session", lambda: None)

    @asynccontextmanager
    async def acquire(key):
        """Model the database lock's nonblocking, per-key exclusion."""
        if key in held_locks:
            yield False
            return
        held_locks.add(key)
        try:
            yield True
        finally:
            held_locks.remove(key)

    locks = MagicMock(acquire=acquire, get_active_locks=AsyncMock(return_value=set()))
    local = LocalStorageImpl(LocalStorageConfig(path=str(tmp_path)))
    ctx = CacheContext(storage=local, lock_manager=locks)

    async def cache(view=None):
        return await cache_file_directly(view or storage, local, "data.jsonl", locks)

    async def status():
        return await get_cache_status_for_files([FileInfo(path="data.jsonl", size=1)], storage, local, locks)

    put(b"A")
    return SimpleNamespace(
        storage=storage,
        objects=objects,
        put=put,
        downloads=downloads,
        client=client,
        ctx=ctx,
        config=config,
        now=now,
        credentials=credentials,
        permissions=permissions,
        cache=cache,
        status=status,
    )


@pytest.fixture
def app(env, monkeypatch):
    """Exercise both HTTP surfaces against the same S3 backend and local cache."""
    app = FastAPI()
    app.include_router(endpoints.router)
    app.include_router(hf_endpoints.router)
    env.config.default_storage_config = env.ctx.storage.config
    app.state.service_configs = {FilesConfig: env.config}
    for dependency in (endpoints.get_entity_client, endpoints.get_nemo_client, endpoints.get_auth_client):
        app.dependency_overrides[dependency] = lambda: MagicMock()
    fileset = SimpleNamespace(
        storage=env.storage.config, name="test", workspace="default", updated_at=None, custom_fields={}
    )
    for module in (endpoints, hf_endpoints):
        monkeypatch.setattr(module, "get_fileset", AsyncMock(return_value=fileset))
        monkeypatch.setattr(module, "resolve_storage_secrets_for_user", AsyncMock(return_value={}))
        monkeypatch.setattr(
            module,
            "storage_impl_factory",
            lambda config, secrets: env.storage if config.type == "s3" else env.ctx.storage,
        )
    monkeypatch.setattr(endpoints, "_get_cache_lock_manager", lambda *_: env.ctx.lock_manager)
    monkeypatch.setattr(hf_endpoints, "FileLockManager", lambda **_: env.ctx.lock_manager)
    return app


@pytest.mark.parametrize(
    "code,detail",
    [
        ("AccessDenied", UPSTREAM_STORAGE_ACCESS_DETAIL),
        ("SlowDown", UPSTREAM_STORAGE_UNAVAILABLE_DETAIL),
        ("InvalidRequest", UPSTREAM_STORAGE_UNAVAILABLE_DETAIL),
    ],
)
async def test_cache_status_head_failure_has_stable_http_error(env, app, code, detail):
    """ListBucket success must not turn a denied/throttled HEAD into an internal 500."""
    env.storage.list_files = AsyncMock(return_value=[FileInfo(path="data.jsonl", size=1)])
    env.client.head_object.side_effect = ClientError(
        {"Error": {"Code": code, "Message": "private upstream detail"}}, "HeadObject"
    )
    url = "/v2/workspaces/default/filesets/test/files"
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app), base_url="http://test") as client:
        assert (await client.get(url)).status_code == 200
        response = await client.get(url, params={"include_cache_status": "true"})
    assert response.status_code == 502
    assert response.json() == {"detail": detail}


@pytest.mark.parametrize("method", ["HEAD", "GET"])
@pytest.mark.parametrize("use_cache", [False, True])
async def test_hf_routes_use_one_snapshot_for_headers_and_body(env, app, method, use_cache):
    """A replacement after HEAD cannot change the selected size or streamed version."""
    if not use_cache:
        env.config.default_storage_config = env.storage.config
    env.put(b"old", {"ETag": '"old"', "VersionId": "old"})
    head = env.client.head_object.side_effect

    async def head_then_replace(**kwargs):
        metadata = await head(**kwargs)
        env.put(b"newer", {"ETag": '"new"', "VersionId": "new"})
        return metadata

    env.client.head_object.side_effect = head_then_replace
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app), base_url="http://test") as client:
        response = await client.request(method, "/v2/hf/default/test/resolve/main/data.jsonl")
    assert response.status_code == 200 and response.headers["content-length"] == "3"
    assert response.content == (b"old" if method == "GET" else b"")
    assert env.client.head_object.await_count == 1
    assert all(request["VersionId"] == "old" for request in env.downloads)


@pytest.mark.parametrize(
    "url",
    [
        "/v2/workspaces/default/filesets/test/-/data.jsonl",
        "/v2/hf/default/test/resolve/main/data.jsonl",
    ],
)
async def test_object_changed_returns_retryable_conflict(env, app, monkeypatch, url):
    """An unversioned overwrite before GET is a conflict, not an upstream outage."""
    download = s3_backend.download_url_streaming

    async def replace_then_download(*args, **kwargs):
        env.put(b"replacement")
        async for chunk in download(*args, **kwargs):
            yield chunk

    monkeypatch.setattr(s3_backend, "download_url_streaming", replace_then_download)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app), base_url="http://test") as client:
        response = await client.get(url)
    assert response.status_code == 409
    assert response.json() == {"detail": "Upstream object changed during download; retry the request"}
    assert not await env.ctx.storage.list_files("cache/s3-v2")


@pytest.mark.parametrize("ttl", [0, 10])
async def test_fileset_http_reads_revalidate_s3_cache(env, app, ttl):
    """PUT/GET/HEAD/DELETE and direct mutations stay coherent, including ranges."""
    env.config.s3_metadata_cache_ttl_seconds = ttl
    await env.ctx.storage.upload("cache/s3/bucket/prefix/data.jsonl", chunks(b"stale"), content_length=5)
    url = "/v2/workspaces/default/filesets/test/-/data.jsonl"
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app), base_url="http://test") as client:
        for content in (b"A", b"B", b"BBBB"):  # Include an equal-size overwrite.
            assert (await client.put(url, content=content)).status_code == 200
            heads = env.client.head_object.await_count
            assert (await client.get(url)).content == content
            assert env.client.head_object.await_count == heads + 1
            count = len(env.downloads)
            assert (await client.get(url)).content == content
            assert len(env.downloads) == count
            assert env.client.head_object.await_count == heads + (1 if ttl else 2)
            assert (await client.head(url)).headers["content-length"] == str(len(content))
        env.put(b"CC")
        if ttl:
            assert (await client.get(url)).content == b"BBBB"
            env.now[0] += ttl
        response = await client.get(url, headers={"Range": "bytes=0-0"})
        assert response.status_code == 206 and response.content == b"C"
        assert response.headers["content-range"] == "bytes 0-0/2"
        assert (await client.delete(url)).status_code == 200
        assert (await client.head(url)).status_code == 404
        assert (await client.get(url)).status_code == 404
        env.put(b"recreated")
        assert (await client.get(url)).content == b"recreated"
        del env.objects["prefix/data.jsonl"]
        env.now[0] += ttl
        assert (await client.get(url)).status_code == 404


async def test_head_get_race_cannot_poison_cache_and_retry_refreshes_metadata(env):
    """A conditional failure leaves no wrong-key bytes and invalidates stale metadata."""
    env.config.s3_metadata_cache_ttl_seconds = 10
    view = await env.storage.for_download("data.jsonl")
    key = await view.get_cache_path_key("data.jsonl")
    env.put(b"BBBB")
    with pytest.raises(StorageObjectChangedError):
        await env.cache(view)
    with pytest.raises(NotFoundError):
        await env.ctx.storage.get_file(key)
    assert (await env.storage.get_file("data.jsonl")).size == 4
    assert env.client.head_object.await_count == 2


@pytest.mark.parametrize("etag", ['"same"', None])
async def test_versioned_download_remains_pinned_after_overwrite(env, etag):
    """Version IDs select the snapshot even when ETags match or are absent."""
    env.put(b"old", {"ETag": etag, "VersionId": "old/version"})
    view = await env.storage.for_download("data.jsonl")
    old_key = await view.get_cache_path_key("data.jsonl")
    env.put(b"newer", {"ETag": etag, "VersionId": "new/version"})
    assert await env.cache(view)
    assert b"".join([chunk async for chunk in await env.ctx.storage.download(old_key, None)]) == b"old"
    assert env.downloads[-1]["VersionId"] == "old/version"
    assert await env.storage.get_cache_path_key("data.jsonl") != old_key
    assert (await view.get_file("data.jsonl")).size == 3


@pytest.mark.parametrize("metadata", [{}, {"VersionId": "null"}])
async def test_missing_validators_bypass_cache(env, metadata):
    """Neither content nor metadata may be reused without an immutable validator."""
    env.config.s3_metadata_cache_ttl_seconds = 10
    env.put(b"A", metadata)
    assert await env.storage.get_cache_path_key("data.jsonl") is None
    tasks = BackgroundTasks()
    data = [chunk async for chunk in download_with_cache(env.storage, "data.jsonl", env.ctx, tasks)]
    assert data == [b"A"] and not tasks.tasks
    env.put(b"BBB", metadata)
    assert (await env.storage.get_file("data.jsonl")).size == 3


async def test_warming_and_status_follow_object_identity(env):
    """A warmed snapshot is cached; a changed object must not inherit that status."""
    env.put(b"A", {"ETag": '"opaque/../etag"', "VersionId": "null"})
    env.storage.list_files = AsyncMock(return_value=[FileInfo(path="data.jsonl", size=1)])
    await warm_fileset_cache(env.storage, env.ctx.storage, env.ctx.lock_manager)
    assert await env.status() == {"data.jsonl": CacheStatus.CACHED}
    assert "VersionId" not in env.downloads[-1]  # A null version requires If-Match instead.
    env.put(b"B")
    assert await env.status() == {"data.jsonl": CacheStatus.NOT_CACHED}


def test_s3_metadata_ttl_config(monkeypatch):
    """Default to ten seconds, support environment overrides, and reject invalid TTLs."""
    monkeypatch.delenv("NHX_FILES_S3_METADATA_CACHE_TTL_SECONDS", raising=False)
    assert FilesConfig().s3_metadata_cache_ttl_seconds == 10
    monkeypatch.setenv("NHX_FILES_S3_METADATA_CACHE_TTL_SECONDS", "2.5")
    assert FilesConfig().s3_metadata_cache_ttl_seconds == 2.5
    monkeypatch.delenv("NHX_FILES_S3_METADATA_CACHE_TTL_SECONDS")
    for value in (-1, float("nan"), float("inf")):
        with pytest.raises(ValueError):
            FilesConfig(s3_metadata_cache_ttl_seconds=value)


async def test_metadata_ttl_is_shared_expires_and_does_not_cache_missing_objects(env):
    """A new backend shares the TTL, but expiry and recreation are visible."""
    env.config.s3_metadata_cache_ttl_seconds = 10
    assert (await env.storage.get_file("data.jsonl")).size == 1
    env.put(b"BBBB")
    other = copy(env.storage)
    for now, size, heads in [(109.9, 1, 1), (110, 4, 2)]:
        env.now[0] = now
        assert (await other.get_file("data.jsonl")).size == size
        assert env.client.head_object.await_count == heads
    del env.objects["prefix/data.jsonl"]
    env.now[0] += 10
    with pytest.raises(NotFoundError):
        await other.for_download("data.jsonl")
    env.put(b"new")
    assert (await other.get_file("data.jsonl")).size == 3


@pytest.mark.parametrize("region", [None, "us-east-1"])
async def test_api_mutations_invalidate_all_credential_and_prefix_aliases(env, region):
    """Local PUT/DELETE invalidate other credential scopes and region/prefix aliases."""
    env.config.s3_metadata_cache_ttl_seconds = 10
    await env.storage.for_download("data.jsonl")
    alias = copy(env.storage)
    alias.config = alias.config.model_copy(update={"prefix": "", "region": region})
    await alias.for_download("prefix/data.jsonl")
    initial_heads = 1 + (region is not None)
    assert env.client.head_object.await_count == initial_heads
    env.credentials.access_key = "other"
    await env.storage.for_download("data.jsonl")
    assert env.client.head_object.await_count == initial_heads + 1
    await env.storage.upload("data.jsonl", chunks(b"BBBB"), content_length=4)
    assert (await alias.get_file("prefix/data.jsonl")).size == 4
    env.credentials.access_key = "test"
    assert (await env.storage.get_file("data.jsonl")).size == 4
    await env.storage.delete("data.jsonl")
    with pytest.raises(NotFoundError):
        await alias.get_file("prefix/data.jsonl")


@pytest.mark.parametrize("ttl", [0, 10])
async def test_failed_refresh_cannot_serve_cached_plaintext(env, ttl):
    """Fresh authorization failures override cached bytes, without caching the error."""
    env.config.s3_metadata_cache_ttl_seconds = ttl
    assert await env.cache()
    env.now[0] += ttl
    head = env.client.head_object.side_effect
    env.client.head_object.side_effect = ClientError({"Error": {"Code": "AccessDenied"}}, "HeadObject")
    with pytest.raises(HTTPException) as exc:
        await resolve_download_storage(env.storage, "data.jsonl")
    assert exc.value.status_code == 502
    env.client.head_object.side_effect = head
    env.put(b"BB")
    assert (await env.storage.get_file("data.jsonl")).size == 2


async def test_metadata_cache_is_bounded_and_credential_scoped(env, monkeypatch):
    """Credential rotation causes a HEAD; LRU eviction bounds retained metadata."""
    env.config.s3_metadata_cache_ttl_seconds = 10
    monkeypatch.setattr(s3_backend, "_METADATA_CACHE_MAX_SIZE", 2)
    for identity in ("first", "second", "third"):
        env.credentials.access_key = identity
        await env.storage.for_download("data.jsonl")
    assert env.client.head_object.await_count == 3
    assert len(s3_backend._metadata_cache) == 2
    assert "upstream-secret" not in repr(s3_backend._metadata_cache)
    env.credentials.access_key = "first"
    await env.storage.for_download("data.jsonl")
    assert env.client.head_object.await_count == 4


async def test_bulk_delete_failure_still_invalidates_metadata(env):
    """A failed bulk delete may have removed objects; conservatively invalidate."""
    env.config.s3_metadata_cache_ttl_seconds = 10
    await env.storage.for_download("data.jsonl")
    paginator = MagicMock()
    paginator.paginate.return_value.__aiter__.return_value = [{"Contents": [{"Key": "prefix/data.jsonl"}]}]
    env.client.get_paginator = MagicMock(return_value=paginator)
    env.client.delete_objects.side_effect = ClientError({"Error": {"Code": "ServiceUnavailable"}}, "DeleteObjects")
    with pytest.raises(S3UnavailableError):
        await env.storage.delete_all()
    env.put(b"BBBB")
    assert (await env.storage.get_file("data.jsonl")).size == 4


async def test_inflight_head_cannot_repopulate_metadata_after_upload(env):
    """A pre-mutation HEAD may serve its caller, but cannot refill the TTL cache."""
    env.config.s3_metadata_cache_ttl_seconds = 10
    started, finish = anyio.Event(), anyio.Event()
    head = env.client.head_object.side_effect

    async def slow_head(**kwargs):
        result = await head(**kwargs)
        started.set()
        await finish.wait()
        return result

    env.client.head_object.side_effect = slow_head
    async with anyio.create_task_group() as tasks:
        tasks.start_soon(env.storage.for_download, "data.jsonl")
        await started.wait()
        await env.storage.upload("data.jsonl", chunks(b"BBBB"), content_length=4)
        finish.set()
    env.client.head_object.side_effect = head
    assert (await env.storage.get_file("data.jsonl")).size == 4
    assert env.client.head_object.await_count == 2


@pytest.mark.parametrize("changed", [False, True])
async def test_getobject_only_credentials_use_guarded_fallback(env, changed):
    """Version-denied reads can use current content only if its ETag still matches."""
    env.put(b"A", {"ETag": '"A"', "VersionId": "version-a"})
    view = await env.storage.for_download("data.jsonl")
    env.permissions["versions"] = False
    if changed:
        env.put(b"BBBB", {"ETag": '"B"', "VersionId": "version-b"})
        with pytest.raises(StorageObjectChangedError):
            await env.cache(view)
    else:
        assert await env.cache(view)
    assert len(env.downloads) == 2
    assert env.downloads[0]["VersionId"] == "version-a"
    assert "VersionId" not in env.downloads[1]
    assert env.downloads[1]["IfMatch"] == '"A"'


async def test_version_denied_without_etag_never_falls_back(env):
    """Without an ETag, retrying the current object could poison a version key."""
    env.put(b"A", {"VersionId": "version-a"})
    env.permissions["versions"] = False
    with pytest.raises(S3BackendError, match="403"):
        await env.cache()
    assert len(env.downloads) == 1


@pytest.mark.parametrize("encryption", ["aws:kms", "aws:kms:dsse"])
async def test_kms_plaintext_cache_cannot_bypass_content_authorization(env, encryption):
    """HEAD permission without kms:Decrypt cannot expose existing cached plaintext."""
    metadata = {"ETag": '"same"', "VersionId": "same-version"}
    env.put(b"secret", metadata)
    assert await env.cache()
    env.put(b"secret", dict(metadata, ServerSideEncryption=encryption))
    env.config.s3_metadata_cache_ttl_seconds = 10
    env.permissions["content"] = False
    env.permissions["versions"] = False
    tasks = BackgroundTasks()
    before = len(env.downloads)
    for _ in range(2):
        with pytest.raises(S3BackendError, match="403"):
            _ = [chunk async for chunk in download_with_cache(env.storage, "data.jsonl", env.ctx, tasks)]
    assert len(env.downloads) == before + 2  # One GET per denied read, not a version probe plus retry.
    assert all("VersionId" not in request for request in env.downloads[before:])
    assert not tasks.tasks and not s3_backend._metadata_cache
    env.permissions["content"] = True
    data = [chunk async for chunk in download_with_cache(env.storage, "data.jsonl", env.ctx, tasks)]
    assert data == [b"secret"] and not tasks.tasks
    assert len(env.downloads) == before + 3
