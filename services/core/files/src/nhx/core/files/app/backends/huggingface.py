# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Huggingface storage backend for Huggingface Hub repositories."""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from typing import TypeVar

import aiohttp
import httpx
from anyio import sleep, to_thread
from huggingface_hub import HfApi, get_hf_file_metadata, hf_hub_url
from huggingface_hub.utils import (
    EntryNotFoundError,
    HfHubHTTPError,
)
from nhx.common.files.storage_config import (
    HuggingfaceStorageConfig as HuggingfaceStorageConfig,
)
from nhx.core.files.app.backends.base import (
    ByteRange,
    FileInfo,
    StorageImpl,
)
from nhx.core.files.app.external_hosts import validate_external_host
from nhx.core.files.app.http_session import get_http_session
from nhx.core.files.app.streaming import download_url_streaming
from nhx.core.files.config import FilesConfig, files_config
from nhx.core.files.exceptions import (
    NotFoundError,
    StorageAccessError,
    StorageBackendError,
    StorageConfigError,
    StorageUnavailableError,
)

logger = logging.getLogger(__name__)

_T = TypeVar("_T")


class HuggingfaceBackendError(StorageBackendError):
    """Raised when there's issues talking to Huggingface."""


class HuggingfaceAccessError(StorageAccessError):
    """Raised when access to a HuggingFace repo is denied (gated, 401, 403)."""


class HuggingfaceConfigError(StorageConfigError):
    """Raised when HuggingFace storage config is invalid (repo/revision not found)."""


class HuggingfaceUnavailableError(StorageUnavailableError):
    """Raised when HuggingFace is unavailable (5xx, 429, timeout)."""


def raise_for_hf_status(
    status_code: int,
    headers: dict[str, str] | None = None,
    url: str | None = None,
) -> None:
    """Raise appropriate HuggingFace exception based on status code and headers.

    Uses the X-Error-Code header that HuggingFace returns to determine the
    specific error type, falling back to status code mapping.

    This code is slightly duplicated from huggingface_hub.util's hf_raise_for_status
    because we use both aiohttp and huggingface_hub for different requests. This
    lets us use this singular helper function for all error-handling throughout this file.
    """
    if status_code < 400:
        return

    error_code = headers.get("X-Error-Code") if headers else None
    context = f" for {url}" if url else ""

    # Map X-Error-Code header to specific exceptions
    if error_code == "GatedRepo":
        raise HuggingfaceAccessError(f"Access denied to gated repository{context}")
    if error_code == "RepoNotFound":
        raise HuggingfaceConfigError(f"Repository not found{context}")
    if error_code == "RevisionNotFound":
        raise HuggingfaceConfigError(f"Revision not found{context}")
    if error_code == "EntryNotFound":
        raise HuggingfaceConfigError(f"Entry not found{context}")

    # Fall back to status code mapping
    if status_code == 401:
        raise HuggingfaceAccessError(f"Unauthorized{context}")
    if status_code == 403:
        raise HuggingfaceAccessError(f"Forbidden{context}")
    if status_code == 404:
        raise HuggingfaceConfigError(f"Not found{context}")
    if status_code == 429:
        raise HuggingfaceUnavailableError(f"Rate limited{context}")
    if status_code >= 500:
        raise HuggingfaceUnavailableError(f"Service error ({status_code}){context}")

    raise HuggingfaceBackendError(f"HTTP {status_code}{context}")


def _map_hf_http_error(exc: HfHubHTTPError) -> Exception:
    """Map HuggingFace Hub HTTP exceptions into storage backend exceptions."""
    if exc.response is not None:
        try:
            raise_for_hf_status(
                exc.response.status_code,
                dict(exc.response.headers),
                str(exc.response.url),
            )
        except (
            HuggingfaceAccessError,
            HuggingfaceConfigError,
            HuggingfaceUnavailableError,
            HuggingfaceBackendError,
        ) as mapped:
            return mapped
    return HuggingfaceBackendError(f"HuggingFace API error: {exc}")


def _ratelimit_reset_seconds(headers: dict[str, str]) -> float | None:
    """Parse HuggingFace's RateLimit t=<seconds-until-reset> field."""
    raw_value = headers.get("RateLimit") or headers.get("ratelimit")
    if not raw_value:
        return None

    for part in raw_value.split(";"):
        key, separator, value = part.strip().partition("=")
        if key != "t" or not separator:
            continue
        try:
            return max(0.0, float(value.split(",", 1)[0].strip().strip('"')))
        except ValueError:
            continue
    return None


def _retry_after_seconds(headers: dict[str, str] | None, status_code: int | None = None) -> float | None:
    """Parse Retry-After, or HF's 429 RateLimit reset, into seconds."""
    if not headers:
        return None

    raw_value = headers.get("Retry-After") or headers.get("retry-after")
    if raw_value:
        try:
            return max(0.0, float(raw_value))
        except ValueError:
            pass

        try:
            retry_at = parsedate_to_datetime(raw_value)
        except (TypeError, ValueError):
            pass
        else:
            if retry_at.tzinfo is None:
                retry_at = retry_at.replace(tzinfo=timezone.utc)
            return max(0.0, (retry_at - datetime.now(timezone.utc)).total_seconds())

    if status_code == 429:
        return _ratelimit_reset_seconds(headers)
    return None


async def _sleep_before_retry(
    *,
    operation: str,
    attempt: int,
    retry_config: FilesConfig,
    headers: dict[str, str] | None,
    status_code: int | None = None,
    error: Exception,
) -> None:
    """Sleep according to Retry-After or exponential backoff before retrying."""
    retry_after = _retry_after_seconds(headers, status_code=status_code)
    delay = (
        min(retry_after, retry_config.hf_retry_max_delay_seconds)
        if retry_after is not None
        else min(
            retry_config.hf_retry_initial_delay_seconds * (2 ** (attempt - 1)),
            retry_config.hf_retry_max_delay_seconds,
        )
    )
    logger.warning(
        "Transient HuggingFace error during %s; retrying attempt %s/%s after %.2fs: %s",
        operation,
        attempt + 1,
        retry_config.hf_retry_attempts,
        delay,
        error,
    )
    await sleep(delay)


async def _run_hf_request(operation: str, request: Callable[[], _T]) -> _T:
    """Run a blocking HuggingFace Hub request with transient retry handling."""
    retry_config = files_config()
    for attempt in range(1, retry_config.hf_retry_attempts + 1):
        try:
            return await to_thread.run_sync(request)
        except EntryNotFoundError:
            raise
        except HfHubHTTPError as exc:
            mapped = _map_hf_http_error(exc)
            if isinstance(mapped, HuggingfaceUnavailableError) and attempt < retry_config.hf_retry_attempts:
                headers = dict(exc.response.headers) if exc.response is not None else None
                status_code = exc.response.status_code if exc.response is not None else None
                await _sleep_before_retry(
                    operation=operation,
                    attempt=attempt,
                    retry_config=retry_config,
                    headers=headers,
                    status_code=status_code,
                    error=mapped,
                )
                continue
            raise mapped from exc
        except httpx.RequestError as exc:
            mapped = HuggingfaceUnavailableError(f"Network error during {operation}: {exc}")
            if attempt < retry_config.hf_retry_attempts:
                await _sleep_before_retry(
                    operation=operation,
                    attempt=attempt,
                    retry_config=retry_config,
                    headers=None,
                    status_code=None,
                    error=mapped,
                )
                continue
            raise mapped from exc

    raise AssertionError("unreachable in _run_hf_request")


def _map_hf_download_error(
    *,
    path: str,
    download_url: str,
    exc: aiohttp.ClientError,
) -> tuple[Exception, dict[str, str] | None]:
    """Map aiohttp download errors and preserve response headers for retry delays."""
    if isinstance(exc, aiohttp.ClientResponseError):
        response_headers = dict(exc.headers) if exc.headers else None
        try:
            raise_for_hf_status(exc.status, response_headers, download_url)
        except (
            HuggingfaceAccessError,
            HuggingfaceConfigError,
            HuggingfaceUnavailableError,
            HuggingfaceBackendError,
        ) as mapped:
            return mapped, response_headers
        return HuggingfaceBackendError(f"HTTP error downloading file {path}: {exc.status}"), response_headers

    return HuggingfaceUnavailableError(f"Network error downloading file {path}: {exc}"), None


async def _map_hf_download_stream_errors(
    stream: AsyncIterator[bytes],
    *,
    path: str,
    download_url: str,
) -> AsyncIterator[bytes]:
    """Translate stream errors after retrying is no longer safe."""
    try:
        async for chunk in stream:
            yield chunk
    except aiohttp.ClientError as exc:
        mapped, _ = _map_hf_download_error(path=path, download_url=download_url, exc=exc)
        raise mapped from exc


async def _stream_hf_download_with_retries(
    *,
    path: str,
    download_url: str,
    session: aiohttp.ClientSession,
    headers: dict[str, str] | None,
    byte_range: ByteRange | None,
    chunk_size: int,
) -> AsyncIterator[bytes]:
    """Stream a download, retrying only errors before the first chunk is emitted."""
    retry_config = files_config()
    for attempt in range(1, retry_config.hf_retry_attempts + 1):
        stream = download_url_streaming(
            url=download_url,
            session=session,
            headers=dict(headers) if headers else None,
            byte_range=byte_range,
            chunk_size=chunk_size,
        )

        try:
            first_chunk = await anext(stream)
        except StopAsyncIteration:
            return
        except aiohttp.ClientError as exc:
            mapped, response_headers = _map_hf_download_error(path=path, download_url=download_url, exc=exc)
            if isinstance(mapped, HuggingfaceUnavailableError) and attempt < retry_config.hf_retry_attempts:
                status_code = exc.status if isinstance(exc, aiohttp.ClientResponseError) else None
                await _sleep_before_retry(
                    operation="download file",
                    attempt=attempt,
                    retry_config=retry_config,
                    headers=response_headers,
                    status_code=status_code,
                    error=mapped,
                )
                continue
            raise mapped from exc

        # Once bytes have reached the caller, retrying would replay the file prefix.
        yield first_chunk
        async for chunk in _map_hf_download_stream_errors(stream, path=path, download_url=download_url):
            yield chunk
        return

    raise AssertionError("unreachable in _stream_hf_download_with_retries")


@dataclass
class HuggingfaceStorageImpl(StorageImpl):
    config: HuggingfaceStorageConfig
    secrets: dict[str, str]
    _api: HfApi = field(init=False)

    def __post_init__(self):
        self._api = HfApi(
            token=self.secrets.get("token"),
            endpoint=self.config.endpoint,
        )

    async def resolve_config(self) -> HuggingfaceStorageConfig:
        """Resolve the revision to a specific commit SHA.

        Queries HuggingFace to get the commit SHA for the configured revision
        (which may be a branch name like 'main' or a tag). Stores the original
        revision for auditing and updates the config with the resolved SHA.

        Returns:
            A new HuggingfaceStorageConfig with the resolved commit SHA.

        Raises:
            HuggingfaceConfigError: If the repository or revision is not found.
        """
        info = await _run_hf_request(
            "resolve repository revision",
            lambda: self._api.repo_info(
                repo_id=self.config.repo_id,
                repo_type=self.config.repo_type,
                revision=self.config.revision,
            ),
        )
        return self.config.model_copy(
            update={
                "original_revision": self.config.revision,
                "revision": info.sha,
            }
        )

    def _get_download_url(self, filepath: str) -> str:
        """Generate a download URL for a file in the Huggingface repo."""
        return hf_hub_url(
            repo_id=self.config.repo_id,
            filename=filepath,
            repo_type=self.config.repo_type,
            revision=self.config.revision,
            endpoint=self.config.endpoint,
        )

    async def _get_hf_file_metadata(self, filepath: str):
        """Get file metadata from Huggingface for a specific file."""
        url = self._get_download_url(filepath)
        return await _run_hf_request(
            "get file metadata",
            lambda: get_hf_file_metadata(url=url, token=self.secrets.get("token")),
        )

    async def list_files(self, path: str | None = None) -> list[FileInfo]:
        """List files in the Huggingface repository."""
        try:
            # list_repo_tree returns RepoFile and RepoFolder objects
            # We filter for files only (items with size attribute)
            items = await _run_hf_request(
                "list repository tree",
                lambda: list(
                    self._api.list_repo_tree(
                        repo_id=self.config.repo_id,
                        repo_type=self.config.repo_type,
                        revision=self.config.revision,
                        path_in_repo=path,
                        recursive=True,
                    )
                ),
            )
        except EntryNotFoundError:
            # list_repo_tree expects a directory path. If path points to a file
            # (not a directory), HuggingFace returns 404. Fall back to checking
            # if it's a single file using get_file.
            if path:
                try:
                    file_info = await self.get_file(path)
                    return [file_info]
                except NotFoundError:
                    # Neither a directory nor a file - return empty list
                    return []
            return []

        file_infos = []
        for item in items:
            # Only include files (RepoFile has size, RepoFolder does not)
            if not hasattr(item, "size") or item.size is None:
                continue
            file_infos.append(
                FileInfo(
                    path=item.path,
                    size=item.size,
                )
            )

        return file_infos

    async def get_file(self, path: str) -> FileInfo:
        """Get metadata for a specific file using Huggingface's file metadata API.

        Override the base class method because list_repo_tree doesn't work
        for individual file paths - it expects directory paths.

        """
        url = self._get_download_url(path)

        try:
            metadata = await _run_hf_request(
                "get file metadata",
                lambda: get_hf_file_metadata(url=url, token=self.secrets.get("token")),
            )
        except EntryNotFoundError as exc:
            raise NotFoundError(f"File '{path}' not found in {self.config.repo_id}@{self.config.revision}") from exc

        return FileInfo(path=path, size=metadata.size)

    async def download(self, path: str, byte_range: ByteRange | None) -> AsyncIterator[bytes]:
        """Download a file from Huggingface.

        This method:
        1. Generates a download URL using hf_hub_url
        2. Downloads the file via HTTP with optional auth header
        3. Streams the content back as an async iterator
        """
        download_url = self._get_download_url(path)

        headers = {}
        if self.secrets.get("token"):
            headers["Authorization"] = f"Bearer {self.secrets.get('token')}"

        async def _download() -> AsyncIterator[bytes]:
            session = get_http_session()
            async for chunk in _stream_hf_download_with_retries(
                path=path,
                download_url=download_url,
                session=session,
                headers=headers if headers else None,
                byte_range=byte_range,
                chunk_size=self.config.read_chunk_size,
            ):
                yield chunk

        return _download()

    async def validate_storage(self):
        """Validate that we can access the Huggingface repository.

        This performs three checks:
        1. Validate endpoint is in the Files service allowed_external_hosts.
        2. Verify the repository exists and is accessible via repo_info.
        3. Verify we can actually download files by getting metadata for a file.

        The third check is important for gated repos where repo_info may succeed
        but file downloads require explicit access approval.
        """
        validate_external_host(self.config.endpoint)
        try:
            repo_info = await _run_hf_request(
                "validate repository",
                lambda: self._api.repo_info(
                    repo_id=self.config.repo_id,
                    repo_type=self.config.repo_type,
                    revision=self.config.revision,
                ),
            )

            # Verify we can actually download files by checking a file's metadata.
            # This catches gated repos where repo_info succeeds but downloads are blocked.
            if repo_info.siblings:
                sibling = repo_info.siblings[0]
                await self._get_hf_file_metadata(sibling.rfilename)

        except (
            HuggingfaceAccessError,
            HuggingfaceConfigError,
            HuggingfaceUnavailableError,
            HuggingfaceBackendError,
        ):
            raise
        except Exception as exc:
            raise HuggingfaceBackendError(
                f"Failed to access Huggingface repository {self.config.repo_id}@{self.config.revision}"
            ) from exc

    async def upload(
        self,
        path: str,
        fstream: AsyncIterator[bytes],
        content_length: int | None = None,
    ) -> FileInfo:
        raise NotImplementedError("Huggingface upload is not implemented")

    async def delete(self, path: str) -> FileInfo:
        raise NotImplementedError("Huggingface delete is not implemented")

    async def get_cache_path_key(self, path: str | None = None) -> str:
        """
        Return cache path that includes repo ID and revision for uniqueness.

        Format: cache/hf/{repo_id}/{revision}/{path}
        This ensures different revisions don't overwrite each other in cache.
        The repo_id naturally creates nested folders (e.g., facebook/opt-125m).

        Args:
            path: File path within the repo. If None, returns the cache root prefix.
        """
        prefix = f"cache/hf/{self.config.repo_id}/{self.config.revision}"
        if path is None:
            return prefix
        return f"{prefix}/{path}"
