# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""S3 storage backend for S3-compatible object storage."""

from __future__ import annotations

import base64
import hashlib
import json
import logging
from collections import OrderedDict
from contextlib import asynccontextmanager
from copy import copy
from dataclasses import dataclass, field, replace
from time import monotonic
from typing import Any, AsyncIterator, NoReturn

import aioboto3
import aiohttp
from botocore.config import Config as BotoConfig
from botocore.exceptions import ClientError
from nhx.common.files.storage_config import S3StorageConfig
from nhx.core.files.app.backends.base import (
    ByteRange,
    FileInfo,
    StorageImpl,
)
from nhx.core.files.app.http_session import get_http_session
from nhx.core.files.app.streaming import download_url_streaming, upload_url_streaming
from nhx.core.files.exceptions import (
    NotFoundError,
    StorageAccessError,
    StorageBackendError,
    StorageConfigError,
    StorageObjectChangedError,
    StorageUnavailableError,
)
from types_aiobotocore_s3 import S3Client

logger = logging.getLogger(__name__)


class S3BackendError(StorageBackendError):
    """Raised when there's issues talking to S3."""


class S3AccessError(StorageAccessError):
    """Raised when access to S3 is denied (invalid credentials, permissions)."""


class S3ConfigError(StorageConfigError):
    """Raised when S3 storage config is invalid (bucket not found, etc.)."""


class S3UnavailableError(StorageUnavailableError):
    """Raised when S3 is unavailable (service errors, throttling)."""


def _raise_for_s3_error(error: ClientError, context: str = "") -> NoReturn:
    """Convert boto3 ClientError to appropriate S3 exception."""
    error_code = error.response.get("Error", {}).get("Code", "")
    error_message = error.response.get("Error", {}).get("Message", str(error))

    ctx = f" {context}" if context else ""

    # Access/auth errors
    if error_code in (
        "AccessDenied",
        "InvalidAccessKeyId",
        "SignatureDoesNotMatch",
        "403",
    ):
        raise S3AccessError(f"Access denied{ctx}: {error_message}")

    # Config errors (resource not found)
    if error_code in ("NoSuchBucket", "NoSuchKey", "NotFound", "404"):
        raise S3ConfigError(f"Not found{ctx}: {error_message}")

    # Service unavailable errors
    if error_code in ("ServiceUnavailable", "SlowDown", "Throttling", "RequestTimeout"):
        raise S3UnavailableError(f"S3 unavailable{ctx}: {error_message}")

    # Default to generic backend error
    raise S3BackendError(f"S3 error{ctx}: {error_message}")


def _add_content_md5_header(params: dict[str, Any], **kwargs: Any) -> None:
    """Add Content-MD5 header for DeleteObjects requests.

    Some S3-compatible providers (Oracle Object Storage, older MinIO versions)
    require the Content-MD5 header for DeleteObjects operations. Botocore 1.36+
    no longer calculates this automatically, using CRC32 checksums instead.

    This event handler manually calculates and adds the MD5 hash to ensure
    compatibility with all S3-compatible backends.

    See: https://github.com/boto/botocore/issues/3415
    """
    body = params.get("body")
    if body:
        if isinstance(body, str):
            body = body.encode("utf-8")
        md5_hash = hashlib.md5(body).digest()
        content_md5 = base64.b64encode(md5_hash).decode("utf-8")
        params["headers"]["Content-MD5"] = content_md5


@dataclass(frozen=True)
class _ObjectSnapshot:
    """Object identity and size shared by one request and its cache writer."""

    path: str
    size: int
    etag: str | None
    version_id: str | None
    kms_encrypted: bool = False


@dataclass(frozen=True)
class _MetadataEntry:
    """A successful HEAD result, aged from the start of its freshness check."""

    checked_at: float
    snapshot: _ObjectSnapshot


# Process-local, bounded LRU. Credential fingerprints isolate upstream access
# scopes; mutation invalidation covers all scopes and overlapping fileset prefixes.
_METADATA_CACHE_MAX_SIZE = 4096
_metadata_cache: OrderedDict[tuple[str, str, str], _MetadataEntry] = OrderedDict()
_metadata_generation = 0


@dataclass
class S3StorageImpl(StorageImpl):
    config: S3StorageConfig
    secrets: dict[str, str]
    _session: aioboto3.Session = field(init=False)
    _snapshot: _ObjectSnapshot | None = field(default=None, init=False)

    def __post_init__(self):
        self._session = aioboto3.Session()
        self._session._session.register("before-call.s3.DeleteObjects", _add_content_md5_header)

    def _get_client_kwargs(self) -> dict[str, Any]:
        """Build kwargs for creating S3 client."""
        kwargs: dict[str, Any] = {
            "config": BotoConfig(signature_version=self.config.signature_version),
        }

        if self.config.region:
            kwargs["region_name"] = self.config.region

        if self.config.endpoint_url:
            kwargs["endpoint_url"] = self.config.endpoint_url

        # Only use explicit credentials if use_sdk_auth is False
        if not self.config.use_sdk_auth:
            access_key = self.secrets.get("access_key_id")
            secret_key = self.secrets.get("secret_access_key")
            if not access_key or not secret_key:
                # This should never happen if S3StorageConfig validation passed,
                # but we check anyway to prevent boto3 from falling back to
                # environment credentials when explicit credentials were intended.
                raise S3ConfigError(
                    "use_sdk_auth=False requires both access_key_id and secret_access_key secrets to be provided."
                )
            kwargs["aws_access_key_id"] = access_key
            kwargs["aws_secret_access_key"] = secret_key

        return kwargs

    def _full_key(self, path: str) -> str:
        """Get the full S3 key including prefix."""
        if self.config.prefix:
            return f"{self.config.prefix.rstrip('/')}/{path}"
        return path

    def _metadata_source(self) -> str:
        """Identify a bucket independently of fileset prefixes and credentials."""
        # Region is a routing/signing hint, not bucket identity. Invalidating
        # across aliases is safe even when they use different region settings.
        return json.dumps([self.config.endpoint_url, self.config.bucket])

    async def _metadata_cache_key(self, path: str) -> tuple[str, str, str] | None:
        """Scope metadata to resolved credentials, without retaining secret text."""
        if self.config.use_sdk_auth:
            credentials = await self._session._session.get_credentials()
            if credentials is None:
                return None
            frozen = await credentials.get_frozen_credentials()
            auth = [frozen.access_key, frozen.secret_key, frozen.token]
        else:
            # Validate required secrets before allowing a metadata cache hit.
            self._get_client_kwargs()
            auth = [self.secrets.get("access_key_id"), self.secrets.get("secret_access_key")]
        # Keep lookups isolated by routing hints (which may select different
        # AWS partitions), while invalidation deliberately spans region aliases.
        fingerprint = hashlib.sha256(json.dumps([auth, self.config.region]).encode()).hexdigest()
        return self._metadata_source(), self._full_key(path), fingerprint

    def _invalidate_metadata(self, path: str | None = None) -> None:
        """Invalidate this object (or prefix), including pending freshness checks."""
        global _metadata_generation
        _metadata_generation += 1
        source = self._metadata_source()
        for key in list(_metadata_cache):
            if key[0] == source and (
                key[1] == self._full_key(path) if path is not None else key[1].startswith(self.config.prefix or "")
            ):
                del _metadata_cache[key]

    @asynccontextmanager
    async def _client(self) -> AsyncIterator[S3Client]:
        """Create an S3 client context manager."""
        async with self._session.client("s3", **self._get_client_kwargs()) as client:
            yield client

    async def list_files(self, path: str | None = None) -> list[FileInfo]:
        """List files in the S3 bucket under the given path.

        The path parameter is used as a prefix filter. S3 prefix matching will
        return all objects whose keys start with the prefix, so:
        - path="dir" matches "dir/file.txt", "dir/subdir/file.txt", etc.
        - path="file.txt" matches "file.txt" (exact match)

        Note: We don't add a trailing slash to the prefix because that would
        prevent matching files (e.g., "file.txt/" won't match "file.txt").
        """
        prefix = self.config.prefix
        if path:
            if prefix:
                prefix = f"{prefix.rstrip('/')}/{path}"
            else:
                prefix = path

        files = []
        async with self._client() as client:
            try:
                paginator = client.get_paginator("list_objects_v2")
                async for page in paginator.paginate(
                    Bucket=self.config.bucket,
                    Prefix=prefix or "",
                ):
                    for obj in page.get("Contents", []):
                        key = obj["Key"]
                        # Make path relative to prefix
                        if self.config.prefix:
                            relative_path = key[len(self.config.prefix.rstrip("/")) + 1 :]
                        else:
                            relative_path = key

                        # Skip "directory" markers (keys ending with /)
                        if relative_path and not relative_path.endswith("/"):
                            files.append(FileInfo(path=relative_path, size=obj["Size"]))
            except ClientError as e:
                _raise_for_s3_error(e, f"listing bucket {self.config.bucket}")

        return files

    async def download(self, path: str, byte_range: ByteRange | None) -> AsyncIterator[bytes]:
        """Download a file from S3 using a presigned URL."""
        full_key = self._full_key(path)
        params: dict[str, Any] = {"Bucket": self.config.bucket, "Key": full_key}
        headers: dict[str, str] = {}
        snapshot = self._snapshot
        if snapshot is not None and snapshot.path == path:
            if snapshot.version_id and not (snapshot.kms_encrypted and snapshot.etag):
                # KMS content is never cached. A conditional current-object GET
                # avoids an unnecessary version-permission probe on every read.
                params["VersionId"] = snapshot.version_id
            if snapshot.etag:
                params["IfMatch"] = snapshot.etag
                headers["If-Match"] = snapshot.etag

        async def presign(request_params: dict[str, Any]) -> str:
            """Sign a GET without fetching content or requiring version permissions."""
            async with self._client() as client:
                try:
                    return await client.generate_presigned_url("get_object", Params=request_params, ExpiresIn=3600)
                except ClientError as e:
                    error_code = e.response.get("Error", {}).get("Code", "")
                    if error_code in ("NoSuchKey", "NotFound", "404"):
                        raise NotFoundError(f"File {path} does not exist in S3 bucket {self.config.bucket}") from e
                    _raise_for_s3_error(e, f"generating presigned URL for {path}")

        url = await presign(params)
        session = get_http_session()

        async def _download() -> AsyncIterator[bytes]:
            current_url = url
            started = False
            for attempt in range(2):
                try:
                    async for chunk in download_url_streaming(
                        current_url,
                        session=session,
                        headers=headers,
                        byte_range=byte_range,
                        chunk_size=self.config.read_chunk_size,
                    ):
                        started = True
                        yield chunk
                    return
                except aiohttp.ClientResponseError as e:
                    if (
                        e.status == 403
                        and attempt == 0
                        and not started
                        and "VersionId" in params
                        and "IfMatch" in params
                    ):
                        # GetObject does not imply GetObjectVersion. Fall back
                        # only with the same ETag guard, never to an unpinned GET.
                        current_url = await presign({k: v for k, v in params.items() if k != "VersionId"})
                        continue
                    if e.status in (404, 412):
                        self._invalidate_metadata(path)
                    if e.status == 412:
                        raise StorageObjectChangedError(
                            "Upstream object changed during download; retry the request"
                        ) from e
                    if e.status == 404:
                        raise NotFoundError(f"File {path} does not exist in S3 bucket {self.config.bucket}") from e
                    raise S3BackendError(f"HTTP error downloading {path}: {e.status}") from e

        return _download()

    async def upload(
        self,
        path: str,
        fstream: AsyncIterator[bytes],
        content_length: int | None = None,
    ) -> FileInfo:
        """Upload a file to S3 using a presigned URL with aiohttp streaming.

        Args:
            path: The path within the storage to upload to
            fstream: Async iterator of bytes to upload
            content_length: Required - the total size of the upload. S3 presigned
                URLs require Content-Length header for PUT requests.
        """
        if content_length is None:
            raise ValueError("content_length is required for S3 uploads")

        full_key = self._full_key(path)

        async with self._client() as client:
            try:
                url = await client.generate_presigned_url(
                    "put_object",
                    Params={"Bucket": self.config.bucket, "Key": full_key},
                    ExpiresIn=3600,
                )
            except ClientError as e:
                _raise_for_s3_error(e, f"generating presigned URL for {path}")

        headers = {"Content-Length": str(content_length)}
        await upload_url_streaming(url, fstream, headers=headers)
        self._invalidate_metadata(path)

        return FileInfo(path=path, size=content_length)

    async def validate_storage(self):
        """Validate that we can access the S3 bucket."""
        async with self._client() as client:
            try:
                # HeadBucket checks existence and permissions
                await client.head_bucket(Bucket=self.config.bucket)
            except ClientError as e:
                _raise_for_s3_error(e, f"validating bucket {self.config.bucket}")

    async def get_file(self, path: str) -> FileInfo:
        """Get file metadata using HeadObject.

        Overrides base class implementation which uses list_files (inefficient
        for S3 and doesn't work correctly due to prefix handling).
        """
        storage = await self.for_download(path)
        if storage._snapshot is None:
            raise S3BackendError("Failed to resolve S3 object metadata")
        return FileInfo(path=path, size=storage._snapshot.size)

    async def for_download(self, path: str) -> S3StorageImpl:
        """Resolve metadata within its TTL and return a pinned download view."""
        if self._snapshot is not None and self._snapshot.path == path:
            return self
        from nhx.core.files.config import files_config

        ttl = files_config().s3_metadata_cache_ttl_seconds
        key = await self._metadata_cache_key(path) if ttl > 0 else None
        checked_at = monotonic()
        if key is not None:
            entry = _metadata_cache.get(key)
            if entry is not None:
                if checked_at - entry.checked_at < ttl:
                    _metadata_cache.move_to_end(key)
                    storage = copy(self)
                    storage._snapshot = replace(entry.snapshot, path=path)
                    return storage
                del _metadata_cache[key]
        generation = _metadata_generation
        async with self._client() as client:
            try:
                head = await client.head_object(Bucket=self.config.bucket, Key=self._full_key(path))
            except ClientError as e:
                error_code = e.response.get("Error", {}).get("Code", "")
                if error_code in ("NoSuchKey", "NotFound", "404"):
                    raise NotFoundError(f"File {path} does not exist in S3 bucket {self.config.bucket}") from e
                _raise_for_s3_error(e, f"getting file info for {path}")
        storage = copy(self)
        version_id = head.get("VersionId")
        storage._snapshot = _ObjectSnapshot(
            path=path,
            size=head["ContentLength"],
            etag=head.get("ETag"),
            # A null version can be overwritten when versioning is suspended.
            version_id=version_id if version_id and version_id != "null" else None,
            kms_encrypted=head.get("ServerSideEncryption") in ("aws:kms", "aws:kms:dsse"),
        )
        if (
            key is not None
            and generation == _metadata_generation
            and (storage._snapshot.etag or storage._snapshot.version_id)
            and not storage._snapshot.kms_encrypted
        ):
            _metadata_cache[key] = _MetadataEntry(checked_at, storage._snapshot)
            _metadata_cache.move_to_end(key)
            while len(_metadata_cache) > _METADATA_CACHE_MAX_SIZE:
                _metadata_cache.popitem(last=False)
        return storage

    async def delete(self, path: str) -> FileInfo:
        """Delete a file from S3."""
        full_key = self._full_key(path)

        async with self._client() as client:
            try:
                # Get file info before deletion
                head = await client.head_object(Bucket=self.config.bucket, Key=full_key)
                file_info = FileInfo(path=path, size=head["ContentLength"])

                # Delete the object
                await client.delete_object(Bucket=self.config.bucket, Key=full_key)
                self._invalidate_metadata(path)

                logger.info(f"Deleted file {path} from S3 bucket {self.config.bucket}")
                return file_info
            except ClientError as e:
                error_code = e.response.get("Error", {}).get("Code", "")
                if error_code in ("NoSuchKey", "NotFound", "404"):
                    raise NotFoundError(f"File {path} does not exist in S3 bucket {self.config.bucket}") from e
                _raise_for_s3_error(e, f"deleting {path}")

    async def delete_all(self) -> None:
        """Delete all files under the prefix in the S3 bucket."""
        # Invalidate before and after: partial bulk deletes can change objects
        # even when a later page fails, and concurrent HEADs must not refill them.
        self._invalidate_metadata()
        prefix = self.config.prefix or ""

        async with self._client() as client:
            try:
                # List and delete all objects with the prefix
                paginator = client.get_paginator("list_objects_v2")
                async for page in paginator.paginate(
                    Bucket=self.config.bucket,
                    Prefix=prefix,
                ):
                    objects = page.get("Contents", [])
                    if objects:
                        delete_request: dict[str, Any] = {"Objects": [{"Key": obj["Key"]} for obj in objects]}
                        await client.delete_objects(
                            Bucket=self.config.bucket,
                            Delete=delete_request,
                        )

                logger.info(f"Deleted all files with prefix '{prefix}' from S3 bucket {self.config.bucket}")
            except ClientError as e:
                _raise_for_s3_error(e, f"deleting all files from bucket {self.config.bucket}")
            finally:
                self._invalidate_metadata()

    async def get_cache_path_key(self, path: str | None = None) -> str | None:
        """Key cached content by source, object key, and freshly resolved identity.

        The new namespace ignores legacy path-only entries. Hashes keep opaque
        ETags, version IDs and object keys safe as local filesystem components.
        """
        source = json.dumps(
            [self.config.endpoint_url, self.config.region, self.config.bucket, self.config.prefix],
            separators=(",", ":"),
        )
        base = f"cache/s3-v2/{hashlib.sha256(source.encode()).hexdigest()}"
        if path is None:
            return base
        storage = await self.for_download(path)
        snapshot = storage._snapshot
        if snapshot is None:
            raise S3BackendError("Failed to resolve S3 object metadata")
        if snapshot.kms_encrypted:
            # HEAD does not prove kms:Decrypt permission. Always GET encrypted
            # content instead of allowing a local plaintext cache to bypass it.
            return None
        if not snapshot.etag and not snapshot.version_id:
            # Nonconforming S3 backends without validators cannot be cached safely.
            return None
        identity = json.dumps([snapshot.version_id, snapshot.etag], separators=(",", ":"))
        object_key = hashlib.sha256(self._full_key(path).encode()).hexdigest()
        version = hashlib.sha256(identity.encode()).hexdigest()
        return f"{base}/{object_key}/{version}"

    def get_duckdb_path(self, path: str) -> str:
        """Return S3 URI for DuckDB access.

        Format: s3://bucket/prefix/path
        """
        prefix = self.config.prefix.rstrip("/") if self.config.prefix else ""
        full_path = f"{prefix}/{path}" if prefix else path
        return f"s3://{self.config.bucket}/{full_path}"
