# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared business logic for Files Service API endpoints."""

import logging
from dataclasses import dataclass
from typing import AsyncIterator

from fastapi import BackgroundTasks, HTTPException, Request
from fastapi.responses import StreamingResponse
from nemo_platform import (
    AsyncNeMoPlatform,
)
from nemo_platform_plugin.client.adapter import client_from_platform
from nemo_platform_plugin.client.errors import NotFoundError as ClientNotFoundError
from nemo_platform_plugin.client.errors import PermissionDeniedError as ClientPermissionDeniedError
from nemo_platform_plugin.secrets.client import AsyncSecretsClient
from nmp.common.auth import AuthClient
from nmp.common.entities.client import EntityClient, EntityNotFoundError
from nmp.common.entities.utils import parse_entity_ref
from nmp.common.observability import MARK_INTERNAL_REQUEST_HEADERS
from nmp.common.secrets.exceptions import SecretAccessDeniedError, SecretNotFoundError
from nmp.core.files.app.backends import FileInfo, StorageConfig
from nmp.core.files.app.backends.base import ByteRange, StorageImpl
from nmp.core.files.app.cache import CacheStatus, cache_file_directly
from nmp.core.files.app.file_lock import FileLockManager
from nmp.core.files.app.range_requests import (
    download_response_status_and_headers,
    parse_range_header,
)
from nmp.core.files.app.streaming import (
    iter_with_inactivity_timeout,
)
from nmp.core.files.entities import Fileset
from nmp.core.files.exceptions import (
    InactivityTimeoutError,
    InvalidPathError,
    InvalidRangeError,
    NotFoundError,
    StorageAccessError,
    StorageBackendError,
    StorageUnavailableError,
)
from starlette.status import (
    HTTP_400_BAD_REQUEST,
    HTTP_404_NOT_FOUND,
    HTTP_416_RANGE_NOT_SATISFIABLE,
    HTTP_502_BAD_GATEWAY,
)

logger = logging.getLogger(__name__)

UPSTREAM_STORAGE_ACCESS_DETAIL = (
    "Failed to access upstream storage for this fileset. Verify that the referenced credentials are valid, "
    "up to date, and have access to the requested resource."
)
UPSTREAM_STORAGE_UNAVAILABLE_DETAIL = "Failed to connect to upstream storage backend"


@dataclass
class CacheContext:
    """Bundles cache storage and lock manager for download caching."""

    storage: StorageImpl
    lock_manager: FileLockManager


async def get_fileset(
    workspace_id: str,
    name: str,
    entity_store: EntityClient,
) -> Fileset:
    """Get a fileset by workspace and name, or raise 404."""
    try:
        return await entity_store.get(Fileset, workspace=workspace_id, name=name)
    except EntityNotFoundError as e:
        raise HTTPException(
            HTTP_404_NOT_FOUND,
            f"Fileset '{name}' not found in workspace '{workspace_id}'",
        ) from e


async def get_file_info(
    storage: StorageImpl,
    path: str,
    fileset_ref: str | None = None,
) -> FileInfo:
    """Get file info or raise 404."""
    try:
        return await storage.get_file(path)
    except NotFoundError as exc:
        detail = f"File '{path}' not found"
        if fileset_ref:
            detail = f"File '{path}' not found in fileset '{fileset_ref}'"
        raise HTTPException(HTTP_404_NOT_FOUND, detail) from exc
    except InvalidPathError as exc:
        raise HTTPException(HTTP_400_BAD_REQUEST, str(exc)) from exc
    except (StorageAccessError, StorageUnavailableError, StorageBackendError) as exc:
        raise runtime_storage_http_error(exc) from exc


async def get_download_file_info(
    storage: StorageImpl,
    path: str,
    fileset_ref: str | None = None,
    cache_ctx: CacheContext | None = None,
) -> FileInfo:
    """Get file info for downloads, preferring the local cache when present."""
    if cache_ctx is not None:
        cache_path_key = await storage.get_cache_path_key(path)
        if cache_path_key is not None:
            try:
                cached_file = await get_file_info(cache_ctx.storage, cache_path_key)
                return FileInfo(path=path, size=cached_file.size)
            except HTTPException as exc:
                if exc.status_code != HTTP_404_NOT_FOUND:
                    raise

    return await get_file_info(storage, path, fileset_ref)


async def list_storage_files(storage: StorageImpl, path: str | None = None) -> list[FileInfo]:
    """List files or raise an HTTP error for runtime storage failures."""
    try:
        return await storage.list_files(path)
    except InvalidPathError as exc:
        raise HTTPException(HTTP_400_BAD_REQUEST, str(exc)) from exc
    except (StorageAccessError, StorageUnavailableError, StorageBackendError) as exc:
        raise runtime_storage_http_error(exc) from exc


def runtime_storage_http_error(exc: Exception) -> HTTPException:
    """Translate runtime storage backend errors to stable HTTP responses."""
    if isinstance(exc, StorageAccessError):
        logger.exception("Runtime storage access error")
        return HTTPException(HTTP_502_BAD_GATEWAY, UPSTREAM_STORAGE_ACCESS_DETAIL)
    if isinstance(exc, (StorageUnavailableError, StorageBackendError)):
        logger.exception("Runtime storage backend unavailable")
        return HTTPException(HTTP_502_BAD_GATEWAY, UPSTREAM_STORAGE_UNAVAILABLE_DETAIL)
    raise exc


def parse_range(range_header: str | None, file_size: int) -> ByteRange | None:
    """Parse Range header or raise 416 Range Not Satisfiable."""
    try:
        return parse_range_header(range_header, file_size=file_size)
    except InvalidRangeError as exc:
        raise HTTPException(
            HTTP_416_RANGE_NOT_SATISFIABLE,
            f"Invalid range: {range_header}",
        ) from exc


async def stream_file_download(
    storage: StorageImpl,
    path: str,
    request: Request,
    file_size: int,
    background_tasks: BackgroundTasks,
    cache_ctx: CacheContext | None = None,
    extra_headers: dict[str, str] | None = None,
) -> StreamingResponse:
    """Stream file content with Range support and optional caching.

    Args:
        storage: Storage backend to download from
        path: File path within the storage
        request: HTTP request (for Range header parsing)
        file_size: Size of the file in bytes
        background_tasks: FastAPI BackgroundTasks for scheduling cache operations
        cache_ctx: Optional cache context for caching external files
        extra_headers: Optional additional response headers
    """
    byte_range = parse_range(request.headers.get("range"), file_size)

    async def stream_with_ctx(source: AsyncIterator[bytes]) -> AsyncIterator[bytes]:
        try:
            async for chunk in source:
                yield chunk
        except Exception:
            # Log the error here because once headers are sent, Starlette can only
            # close the connection. The original exception gets swallowed and replaced
            # with "Response content shorter than Content-Length" from uvicorn.
            logger.exception("Error streaming file download")
            raise

    # Wrap with inactivity timeout and preflight check.
    # preflight=True reads the first chunk immediately (during await) to surface
    # errors before StreamingResponse commits headers.
    try:
        if cache_ctx is not None:
            stream = download_with_cache(storage, path, cache_ctx, background_tasks, byte_range)
        else:
            stream = await storage.download(path, byte_range)
        stream = await iter_with_inactivity_timeout(stream, preflight=True)
    except NotFoundError:
        raise HTTPException(HTTP_404_NOT_FOUND, f"File '{path}' not found")
    except InactivityTimeoutError:
        logger.exception("Timeout during download preflight")
        raise HTTPException(
            HTTP_502_BAD_GATEWAY,
            UPSTREAM_STORAGE_UNAVAILABLE_DETAIL,
        )
    except (StorageAccessError, StorageUnavailableError, StorageBackendError) as exc:
        raise runtime_storage_http_error(exc) from exc
    except Exception:
        logger.exception("Error during download")
        raise HTTPException(
            HTTP_502_BAD_GATEWAY,
            UPSTREAM_STORAGE_UNAVAILABLE_DETAIL,
        )

    stream = stream_with_ctx(stream)

    status, headers = download_response_status_and_headers(byte_range, file_size)
    if extra_headers:
        headers.update(extra_headers)

    return StreamingResponse(
        content=stream,
        status_code=status,
        headers=headers,
        media_type="application/octet-stream",
    )


async def download_with_cache(
    storage: StorageImpl,
    path: str,
    cache_ctx: CacheContext,
    background_tasks: BackgroundTasks,
    byte_range: ByteRange | None = None,
) -> AsyncIterator[bytes]:
    """
    Download a file with caching support.
    On cache miss, schedules background caching and streams directly from source.

    This decouples client download speed from cache write speed - the client always
    gets full download speed from the source, while caching happens independently
    in the background.

    Args:
        storage: Primary storage backend to download from
        path: Path within the storage to download
        cache_ctx: Cache context with storage and lock manager
        background_tasks: FastAPI BackgroundTasks for scheduling cache operations
        byte_range: Optional byte range for partial downloads

    Yields:
        bytes chunks of file content
    """
    cache_path_key = await storage.get_cache_path_key(path)

    # Try to serve from cache if cacheable
    if cache_path_key is not None:
        try:
            async for chunk in await cache_ctx.storage.download(cache_path_key, byte_range):
                yield chunk
            logger.debug("Cache hit", extra={"cache_path_key": cache_path_key})
            return
        except NotFoundError:
            # Cache miss - schedule background caching of full file
            logger.info("Cache miss, scheduling background cache", extra={"path": path})
            background_tasks.add_task(
                cache_file_directly,
                storage,
                cache_ctx.storage,
                path,
                cache_ctx.lock_manager,
            )

    # Stream from source (either not cacheable, or cache miss)
    async for chunk in await storage.download(path, byte_range):
        yield chunk


async def resolve_storage_secrets(storage: StorageConfig, workspace: str, sdk: AsyncNeMoPlatform) -> dict[str, str]:
    """Resolve all secret references in a storage config."""
    secrets: dict[str, str] = {}
    secrets_client = client_from_platform(sdk, AsyncSecretsClient)
    for key, secret_ref in storage.get_secret_references().items():
        parsed = parse_entity_ref(secret_ref.root, workspace)
        try:
            response = (await secrets_client.access_secret(name=parsed.name, workspace=parsed.workspace)).data()
            secrets[key] = response.value
        except ClientNotFoundError:
            raise SecretNotFoundError(
                f"Secret '{parsed.workspace}/{parsed.name}' not found",
            )
        except ClientPermissionDeniedError:
            raise SecretAccessDeniedError(
                f"Access denied to secret '{parsed.workspace}/{parsed.name}'",
            )
    return secrets


async def resolve_storage_secrets_for_user(
    storage: StorageConfig,
    workspace: str,
    sdk: AsyncNeMoPlatform,
    auth_client: AuthClient,
) -> dict[str, str]:
    """Resolve storage secrets using delegated headers on request-scoped SDK."""
    effective_principal = auth_client.principal.effective_principal
    headers = {
        **MARK_INTERNAL_REQUEST_HEADERS,
        "X-NMP-Principal-Id": "service:files",
        "X-NMP-Principal-On-Behalf-Of": effective_principal.id,
    }
    if effective_principal.email:
        headers["X-NMP-Principal-On-Behalf-Of-Email"] = effective_principal.email
    if effective_principal.groups:
        headers["X-NMP-Principal-On-Behalf-Of-Groups"] = ",".join(effective_principal.groups)
    service_sdk = sdk.with_options(set_default_headers=headers)
    return await resolve_storage_secrets(storage, workspace, service_sdk)


async def get_cache_status_for_files(
    files: list[FileInfo],
    source_storage: StorageImpl,
    cache_storage: StorageImpl,
    lock_manager: FileLockManager,
) -> dict[str, CacheStatus]:
    """Determine cache status for each file.

    Args:
        files: List of files to check cache status for
        source_storage: The external storage backend (HuggingFace, NGC)
        cache_storage: The cache storage backend (default/local storage)
        lock_manager: Lock manager for checking active caching operations

    Returns:
        Mapping of file path -> cache status.
        For default storage (uncacheable), returns empty dict.
    """
    # Get cache root prefix for this source
    cache_prefix = await source_storage.get_cache_path_key()
    if not cache_prefix or not files:
        return {}

    # Generate cache path keys for each file
    cache_paths: dict[str, str] = {}
    for f in files:
        cache_key = await source_storage.get_cache_path_key(f.path)
        if cache_key:
            cache_paths[f.path] = cache_key

    if not cache_paths:
        return {}

    # List only files under this source's cache prefix
    cached_files = await cache_storage.list_files(cache_prefix)
    cached_keys = {f.path for f in cached_files}

    # Only query locks for files that aren't cached
    uncached_keys = [ck for ck in cache_paths.values() if ck not in cached_keys]
    active_locks = await lock_manager.get_active_locks(uncached_keys) if uncached_keys else set()

    # Determine status for each file
    result: dict[str, CacheStatus] = {}
    for file_path, cache_key in cache_paths.items():
        if cache_key in cached_keys:
            result[file_path] = CacheStatus.CACHED
        elif cache_key in active_locks:
            result[file_path] = CacheStatus.CACHING
        else:
            result[file_path] = CacheStatus.NOT_CACHED
    return result
