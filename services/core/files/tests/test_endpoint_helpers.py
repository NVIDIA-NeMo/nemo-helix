# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for endpoint helper functions."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from nemo_platform import AsyncNeMoPlatform, Omit
from nemo_platform_plugin.client.errors import NotFoundError as ClientNotFoundError
from nmp.common.api.common import SecretRef
from nmp.common.auth import AuthClient, Principal
from nmp.common.config import AuthConfig
from nmp.common.observability import MARK_INTERNAL_REQUEST_HEADERS
from nmp.common.secrets.exceptions import SecretNotFoundError
from nmp.core.files.api.endpoint_helpers import (
    CacheContext,
    get_cache_status_for_files,
    get_download_file_info,
    get_file_info,
    list_storage_files,
    resolve_storage_secrets,
    resolve_storage_secrets_for_user,
    stream_file_download,
)
from nmp.core.files.app.backends.base import FileInfo
from nmp.core.files.app.backends.huggingface import HuggingfaceStorageConfig
from nmp.core.files.app.backends.local import LocalStorageConfig
from nmp.core.files.app.backends.ngc import NGCStorageConfig
from nmp.core.files.app.cache import CacheStatus
from nmp.core.files.exceptions import NotFoundError, StorageAccessError


@pytest.fixture
def mock_sdk():
    """Mock the platform SDK and the SecretsClient it is adapted into.

    ``resolve_storage_secrets`` now wraps the SDK with
    ``client_from_platform(sdk, AsyncSecretsClient)`` and calls
    ``access_secret(...).data()``, so we patch the adapter to return a mock
    secrets client and drive that. ``mock_sdk.access_secret`` is the AsyncMock
    whose ``.return_value`` / ``.side_effect`` the tests set; the returned
    object's ``.data()`` yields the response model.
    """
    secrets_client = MagicMock()
    secrets_client.access_secret = AsyncMock()
    with (
        patch("nmp.common.sdk_factory.get_async_platform_sdk") as mock,
        patch("nmp.core.files.api.endpoint_helpers.client_from_platform", return_value=secrets_client),
    ):
        sdk = MagicMock()
        mock.return_value = sdk
        # Expose the secrets-client mock as the handle tests configure/assert on.
        sdk.access_secret = secrets_client.access_secret
        yield sdk


def _access_result(value: str) -> MagicMock:
    """Build an object mimicking ``NemoResponse`` — ``.data().value``."""
    resp = MagicMock()
    resp.data.return_value = MagicMock(value=value)
    return resp


async def test_resolve_hf_storage_with_token(mock_sdk):
    """Test resolving secrets for HuggingFace storage with token_secret."""
    mock_sdk.access_secret.return_value = _access_result("hf_token_value")

    config = HuggingfaceStorageConfig(
        repo_id="org/repo",
        token_secret=SecretRef(root="my-hf-token"),
    )

    secrets = await resolve_storage_secrets(config, "my-workspace", mock_sdk)

    assert secrets == {"token": "hf_token_value"}
    mock_sdk.access_secret.assert_called_once_with(name="my-hf-token", workspace="my-workspace")


async def test_resolve_hf_storage_without_token(mock_sdk):
    """Test resolving secrets for HuggingFace storage without token (public repo)."""
    config = HuggingfaceStorageConfig(repo_id="public-org/public-repo")

    secrets = await resolve_storage_secrets(config, "default", mock_sdk)

    assert secrets == {}
    mock_sdk.access_secret.assert_not_called()


async def test_resolve_local_storage(mock_sdk):
    """Test resolving secrets for local storage (no secrets needed)."""
    config = LocalStorageConfig(path="/data/filesets/my-fileset")

    secrets = await resolve_storage_secrets(config, "default", mock_sdk)

    assert secrets == {}
    mock_sdk.access_secret.assert_not_called()


async def test_resolve_ngc_storage(mock_sdk):
    """Test resolving secrets for NGC storage with qualified secret ref (workspace/name)."""
    mock_sdk.access_secret.return_value = _access_result("ngc_api_key_value")

    # Use qualified format: shared-workspace/shared-ngc-key
    config = NGCStorageConfig(
        org="nvidia",
        team="my-team",
        target="my-model",
        api_key_secret=SecretRef(root="shared-workspace/shared-ngc-key"),
    )

    secrets = await resolve_storage_secrets(config, "prod-workspace", mock_sdk)

    assert secrets == {"api_key": "ngc_api_key_value"}
    # Should use workspace from the qualified ref, not the default
    mock_sdk.access_secret.assert_called_once_with(name="shared-ngc-key", workspace="shared-workspace")


async def test_resolve_storage_secrets_propagates_not_found(mock_sdk):
    """A 404 from the secrets service is mapped to SecretNotFoundError."""
    mock_sdk.access_secret.side_effect = ClientNotFoundError(MagicMock(status_code=404))

    config = HuggingfaceStorageConfig(
        repo_id="org/repo",
        token_secret=SecretRef(root="nonexistent-secret"),
    )

    with pytest.raises(SecretNotFoundError):
        await resolve_storage_secrets(config, "default", mock_sdk)


async def test_resolve_storage_secrets_for_user_delegates_effective_principal_claims():
    config = HuggingfaceStorageConfig(
        repo_id="org/repo",
        token_secret=SecretRef(root="my-hf-token"),
    )
    auth_client = AuthClient(
        config=AuthConfig(),
        principal=Principal(
            id="service:jobs",
            groups=["system:serviceaccounts"],
            on_behalf_of="creator@example.com",
            on_behalf_of_email="creator@example.com",
            on_behalf_of_groups=["workspace-editors", "ml-team"],
        ),
    )
    sdk = AsyncNeMoPlatform(base_url="http://testserver")
    captured_headers: dict[str, str | Omit] = {}
    secrets_client = MagicMock()
    secrets_client.access_secret = AsyncMock(return_value=_access_result("hf_token_value"))

    def capture_service_sdk(service_sdk: AsyncNeMoPlatform, _client_type: type[object]) -> MagicMock:
        captured_headers.update(service_sdk.default_headers)
        return secrets_client

    try:
        with patch("nmp.core.files.api.endpoint_helpers.client_from_platform", side_effect=capture_service_sdk):
            secrets = await resolve_storage_secrets_for_user(config, "my-workspace", sdk, auth_client)
    finally:
        await sdk.close()

    assert secrets == {"token": "hf_token_value"}
    for header, value in MARK_INTERNAL_REQUEST_HEADERS.items():
        assert captured_headers[header] == value
    assert captured_headers["X-NMP-Principal-Id"] == "service:files"
    assert captured_headers["X-NMP-Principal-On-Behalf-Of"] == "creator@example.com"
    assert captured_headers["X-NMP-Principal-On-Behalf-Of-Email"] == "creator@example.com"
    assert captured_headers["X-NMP-Principal-On-Behalf-Of-Groups"] == "workspace-editors,ml-team"


# Tests for get_cache_status_for_files


@pytest.fixture
def mock_source_storage():
    """Mock source storage backend (external storage like HuggingFace)."""
    storage = MagicMock()
    # By default, simulate HuggingFace-like cache path keys
    storage.get_cache_path_key = AsyncMock(
        side_effect=lambda path=None: "cache/hf/org/repo/main" if path is None else f"cache/hf/org/repo/main/{path}"
    )
    return storage


@pytest.fixture
def mock_cache_storage():
    """Mock cache storage backend (local storage)."""
    storage = AsyncMock()
    storage.list_files.return_value = []
    return storage


@pytest.fixture
def mock_lock_manager():
    """Mock FileLockManager."""
    lock_manager = AsyncMock()
    lock_manager.get_active_locks.return_value = set()
    return lock_manager


async def test_get_cache_status_for_files_uncacheable_storage(mock_cache_storage, mock_lock_manager):
    """Test get_cache_status_for_files returns empty dict for uncacheable storage."""
    # Local storage returns None for cache path key (both with and without args)
    mock_source = MagicMock()
    mock_source.get_cache_path_key = AsyncMock(return_value=None)

    files = [FileInfo(path="file1.txt", size=100)]
    result = await get_cache_status_for_files(files, mock_source, mock_cache_storage, mock_lock_manager)

    assert result == {}
    mock_cache_storage.list_files.assert_not_called()
    mock_lock_manager.get_active_locks.assert_not_called()


async def test_get_cache_status_for_files_all_cached(mock_source_storage, mock_cache_storage, mock_lock_manager):
    """Test get_cache_status_for_files when all files are cached."""
    files = [
        FileInfo(path="file1.txt", size=100),
        FileInfo(path="file2.txt", size=200),
    ]

    # Cache storage has both files
    mock_cache_storage.list_files.return_value = [
        FileInfo(path="cache/hf/org/repo/main/file1.txt", size=100),
        FileInfo(path="cache/hf/org/repo/main/file2.txt", size=200),
    ]

    result = await get_cache_status_for_files(files, mock_source_storage, mock_cache_storage, mock_lock_manager)

    assert result == {
        "file1.txt": CacheStatus.CACHED,
        "file2.txt": CacheStatus.CACHED,
    }
    # Should not query locks when all files are cached
    mock_lock_manager.get_active_locks.assert_not_called()


async def test_get_cache_status_for_files_none_cached(mock_source_storage, mock_cache_storage, mock_lock_manager):
    """Test get_cache_status_for_files when no files are cached."""
    files = [
        FileInfo(path="file1.txt", size=100),
        FileInfo(path="file2.txt", size=200),
    ]

    # Cache storage is empty
    mock_cache_storage.list_files.return_value = []
    mock_lock_manager.get_active_locks.return_value = set()

    result = await get_cache_status_for_files(files, mock_source_storage, mock_cache_storage, mock_lock_manager)

    assert result == {
        "file1.txt": CacheStatus.NOT_CACHED,
        "file2.txt": CacheStatus.NOT_CACHED,
    }


async def test_get_cache_status_for_files_mixed_status(mock_source_storage, mock_cache_storage, mock_lock_manager):
    """Test get_cache_status_for_files with mix of cached, caching, and not_cached."""
    files = [
        FileInfo(path="cached.txt", size=100),
        FileInfo(path="caching.txt", size=200),
        FileInfo(path="not_cached.txt", size=300),
    ]

    # Only cached.txt is in cache
    mock_cache_storage.list_files.return_value = [
        FileInfo(path="cache/hf/org/repo/main/cached.txt", size=100),
    ]

    # caching.txt has an active lock
    mock_lock_manager.get_active_locks.return_value = {"cache/hf/org/repo/main/caching.txt"}

    result = await get_cache_status_for_files(files, mock_source_storage, mock_cache_storage, mock_lock_manager)

    assert result == {
        "cached.txt": CacheStatus.CACHED,
        "caching.txt": CacheStatus.CACHING,
        "not_cached.txt": CacheStatus.NOT_CACHED,
    }


async def test_get_cache_status_for_files_empty_list(mock_source_storage, mock_cache_storage, mock_lock_manager):
    """Test get_cache_status_for_files with empty file list."""
    result = await get_cache_status_for_files([], mock_source_storage, mock_cache_storage, mock_lock_manager)

    assert result == {}


async def test_get_cache_status_for_files_uses_cache_prefix(mock_source_storage, mock_cache_storage, mock_lock_manager):
    """Test get_cache_status_for_files uses cache prefix for listing."""
    files = [FileInfo(path="file.txt", size=100)]
    mock_cache_storage.list_files.return_value = []

    await get_cache_status_for_files(files, mock_source_storage, mock_cache_storage, mock_lock_manager)

    # Should list files using the cache prefix
    mock_cache_storage.list_files.assert_called_once_with("cache/hf/org/repo/main")


# Tests for stream_file_download preflight handling


@pytest.fixture
def mock_storage():
    """Mock storage backend for download tests."""
    storage = AsyncMock()
    return storage


@pytest.fixture
def mock_request():
    """Mock FastAPI request."""
    request = MagicMock()
    request.headers = {}
    return request


@pytest.fixture
def mock_background_tasks():
    """Mock FastAPI BackgroundTasks."""
    return MagicMock()


async def test_stream_file_download_preflight_not_found_error(mock_storage, mock_request, mock_background_tasks):
    """Test that NotFoundError during preflight is converted to HTTP 404."""
    from fastapi import HTTPException
    from nmp.core.files.api.endpoint_helpers import stream_file_download
    from nmp.core.files.exceptions import NotFoundError
    from starlette.status import HTTP_404_NOT_FOUND

    async def error_on_first_chunk():
        raise NotFoundError("File not found in storage")
        yield  # pragma: no cover

    mock_storage.download.return_value = error_on_first_chunk()

    with pytest.raises(HTTPException) as exc_info:
        await stream_file_download(
            storage=mock_storage,
            path="missing.txt",
            request=mock_request,
            file_size=100,
            background_tasks=mock_background_tasks,
        )

    assert exc_info.value.status_code == HTTP_404_NOT_FOUND
    assert "missing.txt" in exc_info.value.detail


async def test_stream_file_download_preflight_connection_error(mock_storage, mock_request, mock_background_tasks):
    """Test that connection errors during preflight are converted to HTTP 502."""
    from fastapi import HTTPException
    from nmp.core.files.api.endpoint_helpers import stream_file_download
    from starlette.status import HTTP_502_BAD_GATEWAY

    async def error_on_first_chunk():
        raise ConnectionError("Network failure")
        yield  # pragma: no cover

    mock_storage.download.return_value = error_on_first_chunk()

    with pytest.raises(HTTPException) as exc_info:
        await stream_file_download(
            storage=mock_storage,
            path="file.txt",
            request=mock_request,
            file_size=100,
            background_tasks=mock_background_tasks,
        )

    assert exc_info.value.status_code == HTTP_502_BAD_GATEWAY
    assert "storage backend" in exc_info.value.detail.lower()


async def test_get_file_info_storage_access_error_returns_generic_502():
    """Test that runtime storage auth failures are normalized to a stable 502."""
    from fastapi import HTTPException
    from starlette.status import HTTP_502_BAD_GATEWAY

    storage = AsyncMock()
    storage.get_file.side_effect = StorageAccessError("Access denied to gated repository")

    with patch("nmp.core.files.api.endpoint_helpers.logger.exception") as mock_log:
        with pytest.raises(HTTPException) as exc_info:
            await get_file_info(storage, "config.json", "default/my-fileset")

    assert exc_info.value.status_code == HTTP_502_BAD_GATEWAY
    assert "referenced credentials are valid" in exc_info.value.detail
    mock_log.assert_called_once_with("Runtime storage access error")


async def test_get_download_file_info_prefers_cached_file_size():
    """Test that cached downloads use the cache storage for file size."""
    source_storage = AsyncMock()
    source_storage.get_cache_path_key.return_value = "cache/ngc/path/file.bin"

    cache_storage = AsyncMock()
    cache_storage.get_file.return_value = FileInfo(path="cache/ngc/path/file.bin", size=1234)

    cache_ctx = CacheContext(storage=cache_storage, lock_manager=AsyncMock())

    result = await get_download_file_info(
        source_storage,
        "file.bin",
        "default/my-fileset",
        cache_ctx=cache_ctx,
    )

    assert result == FileInfo(path="file.bin", size=1234)
    cache_storage.get_file.assert_called_once_with("cache/ngc/path/file.bin")
    source_storage.get_file.assert_not_called()


async def test_get_download_file_info_falls_back_to_source_on_cache_miss():
    """Test that cache misses still fetch metadata from the source backend."""
    source_storage = AsyncMock()
    source_storage.get_cache_path_key.return_value = "cache/ngc/path/file.bin"
    source_storage.get_file.return_value = FileInfo(path="file.bin", size=4321)

    cache_storage = AsyncMock()
    cache_storage.get_file.side_effect = NotFoundError("not cached")

    cache_ctx = CacheContext(storage=cache_storage, lock_manager=AsyncMock())

    result = await get_download_file_info(
        source_storage,
        "file.bin",
        "default/my-fileset",
        cache_ctx=cache_ctx,
    )

    assert result == FileInfo(path="file.bin", size=4321)
    cache_storage.get_file.assert_called_once_with("cache/ngc/path/file.bin")
    source_storage.get_file.assert_called_once_with("file.bin")


async def test_get_download_file_info_non_404_cache_error_is_raised():
    """Test that non-404 cache metadata errors are surfaced instead of falling back."""
    from fastapi import HTTPException
    from starlette.status import HTTP_502_BAD_GATEWAY

    source_storage = AsyncMock()
    source_storage.get_cache_path_key.return_value = "cache/ngc/path/file.bin"

    cache_storage = AsyncMock()
    cache_storage.get_file.side_effect = StorageAccessError("cache unavailable")

    cache_ctx = CacheContext(storage=cache_storage, lock_manager=AsyncMock())

    with pytest.raises(HTTPException) as exc_info:
        await get_download_file_info(
            source_storage,
            "file.bin",
            "default/my-fileset",
            cache_ctx=cache_ctx,
        )

    assert exc_info.value.status_code == HTTP_502_BAD_GATEWAY
    source_storage.get_file.assert_not_called()


async def test_list_storage_files_storage_access_error_returns_generic_502():
    """Test that list operations use the same runtime storage error mapping."""
    from fastapi import HTTPException
    from starlette.status import HTTP_502_BAD_GATEWAY

    storage = AsyncMock()
    storage.list_files.side_effect = StorageAccessError("Unauthorized")

    with pytest.raises(HTTPException) as exc_info:
        await list_storage_files(storage, "subdir")

    assert exc_info.value.status_code == HTTP_502_BAD_GATEWAY
    assert "referenced credentials are valid" in exc_info.value.detail


async def test_stream_file_download_preflight_storage_access_error_returns_generic_502(
    mock_storage, mock_request, mock_background_tasks
):
    """Test that auth failures during preflight return the normalized upstream auth message."""
    from fastapi import HTTPException
    from starlette.status import HTTP_502_BAD_GATEWAY

    async def error_on_first_chunk():
        raise StorageAccessError("Unauthorized")
        yield  # pragma: no cover

    mock_storage.download.return_value = error_on_first_chunk()

    with pytest.raises(HTTPException) as exc_info:
        await stream_file_download(
            storage=mock_storage,
            path="file.txt",
            request=mock_request,
            file_size=100,
            background_tasks=mock_background_tasks,
        )

    assert exc_info.value.status_code == HTTP_502_BAD_GATEWAY
    assert "referenced credentials are valid" in exc_info.value.detail


async def test_stream_file_download_setup_storage_access_error_returns_generic_502(
    mock_storage, mock_request, mock_background_tasks
):
    """Test that auth failures creating the download stream are normalized too."""
    from fastapi import HTTPException
    from starlette.status import HTTP_502_BAD_GATEWAY

    mock_storage.download.side_effect = StorageAccessError("Unauthorized")

    with pytest.raises(HTTPException) as exc_info:
        await stream_file_download(
            storage=mock_storage,
            path="file.txt",
            request=mock_request,
            file_size=100,
            background_tasks=mock_background_tasks,
        )

    assert exc_info.value.status_code == HTTP_502_BAD_GATEWAY
    assert "referenced credentials are valid" in exc_info.value.detail


async def test_stream_file_download_preflight_success_returns_streaming_response(
    mock_storage, mock_request, mock_background_tasks
):
    """Test that successful preflight returns StreamingResponse with all chunks."""
    from fastapi.responses import StreamingResponse
    from nmp.core.files.api.endpoint_helpers import stream_file_download

    async def mock_download():
        yield b"chunk1"
        yield b"chunk2"

    mock_storage.download.return_value = mock_download()

    response = await stream_file_download(
        storage=mock_storage,
        path="file.txt",
        request=mock_request,
        file_size=12,
        background_tasks=mock_background_tasks,
    )

    assert isinstance(response, StreamingResponse)

    # Consume the response body to verify all chunks are yielded
    chunks = []
    async for chunk in response.body_iterator:
        if isinstance(chunk, bytes):
            chunks.append(chunk)
        elif isinstance(chunk, str):
            chunks.append(chunk.encode())
        else:
            chunks.append(bytes(chunk))
    assert b"".join(chunks) == b"chunk1chunk2"
