# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""S3 storage backend for S3-compatible object storage."""

from __future__ import annotations

import base64
import hashlib
import logging
from concurrent.futures import CancelledError
from contextlib import asynccontextmanager, closing
from dataclasses import dataclass, field
from io import BufferedReader, RawIOBase
from typing import Any, AsyncIterator, NoReturn

import aioboto3
import aiohttp
import boto3
from anyio import create_task_group, from_thread, to_thread
from boto3.s3.transfer import TransferConfig
from botocore.config import Config as BotoConfig
from botocore.exceptions import ClientError
from nhx.common.files.storage_config import S3StorageConfig
from nhx.core.files.app.backends.base import (
    ByteRange,
    FileInfo,
    StorageImpl,
)
from nhx.core.files.app.http_session import get_http_session
from nhx.core.files.app.streaming import download_url_streaming
from nhx.core.files.exceptions import (
    NotFoundError,
    StorageAccessError,
    StorageBackendError,
    StorageConfigError,
    StorageUnavailableError,
)
from s3transfer.utils import ChunksizeAdjuster
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
        "ExpiredToken",
        "InvalidToken",
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


class _UploadStream(RawIOBase):
    """Expose the async request stream to boto3's non-threaded transfer manager."""

    def __init__(self, stream: AsyncIterator[bytes], content_length: int):
        self.stream = stream
        self.remaining = content_length
        self.pending = memoryview(b"")

    def readable(self) -> bool:
        return True

    def readinto(self, buffer: Any) -> int:
        from_thread.check_cancelled()
        while not self.pending:
            try:
                chunk = from_thread.run(anext, self.stream, None)
            except CancelledError:
                from_thread.check_cancelled()
                raise
            if chunk is None:
                if self.remaining:
                    raise ValueError("Upload body does not match Content-Length")
                return 0
            self.remaining -= len(chunk)
            if self.remaining < 0:
                raise ValueError("Upload body exceeds Content-Length")
            self.pending = memoryview(chunk)
        size = min(len(buffer), len(self.pending))
        buffer[:size] = self.pending[:size]
        self.pending = self.pending[size:]
        return size


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


@dataclass
class S3StorageImpl(StorageImpl):
    config: S3StorageConfig
    secrets: dict[str, str]
    _session: aioboto3.Session = field(init=False)

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

    def _storage_boundary(self) -> str:
        """Get the S3 key prefix that scopes this fileset's directory.

        For a configured prefix ``foo`` this is ``foo/``; for ``foo/`` it stays
        ``foo/`` (``_full_key`` already strips one trailing slash). For an empty
        prefix it is empty, meaning the whole bucket. Keys under this boundary
        (plus the prefix object itself) belong to this fileset; raw string
        prefix neighbors such as ``foo-bar`` or ``foo.txt`` do not.
        """
        return self._full_key("")

    def _owns_key(self, key: str) -> bool:
        """Whether an S3 key belongs to this fileset's storage.

        S3 prefix matching is a raw string match, so a listing for ``foo`` also
        returns ``foo-bar`` and ``foo.txt``. A key is owned only when it equals
        the configured prefix (the file-like object itself) or starts with the
        directory boundary (``foo/``).
        """
        return key == self.config.prefix or key.startswith(self._storage_boundary())

    @asynccontextmanager
    async def _client(self) -> AsyncIterator[S3Client]:
        """Create an S3 client context manager."""
        async with self._session.client("s3", **self._get_client_kwargs()) as client:
            yield client

    async def list_files(self, path: str | None = None) -> list[FileInfo]:
        """List files in the S3 bucket under the given path.

        The query prefix is the fileset's storage boundary (``prefix/``) plus
        the optional path filter, so a fileset stored at ``foo`` never lists
        objects of the neighboring ``foo-bar`` fileset. The path filter itself
        is a raw S3 prefix match within the fileset:
        - path="dir" matches "dir/file.txt", "dir/subdir/file.txt", etc.
        - path="file.txt" matches "file.txt", "file.txt/a.txt", and "file.txt.bak"

        Note: We don't add a trailing slash to the path filter because that
        would prevent matching files (e.g., "file.txt/" won't match "file.txt").
        """
        boundary = self._storage_boundary()
        prefix = boundary + (path or "")

        files = []
        async with self._client() as client:
            try:
                paginator = client.get_paginator("list_objects_v2")
                async for page in paginator.paginate(
                    Bucket=self.config.bucket,
                    Prefix=prefix,
                ):
                    for obj in page.get("Contents", []):
                        key = obj["Key"]
                        # Make path relative to the fileset's storage boundary
                        relative_path = key[len(boundary) :]

                        # Skip "directory" markers (keys ending with /)
                        if relative_path and not relative_path.endswith("/"):
                            files.append(FileInfo(path=relative_path, size=obj["Size"]))
            except ClientError as e:
                _raise_for_s3_error(e, f"listing bucket {self.config.bucket}")

        return files

    async def download(self, path: str, byte_range: ByteRange | None) -> AsyncIterator[bytes]:
        """Download a file from S3 using a presigned URL."""
        full_key = self._full_key(path)

        async with self._client() as client:
            try:
                url = await client.generate_presigned_url(
                    "get_object",
                    Params={"Bucket": self.config.bucket, "Key": full_key},
                    ExpiresIn=3600,
                )
            except ClientError as e:
                error_code = e.response.get("Error", {}).get("Code", "")
                if error_code in ("NoSuchKey", "NotFound", "404"):
                    raise NotFoundError(f"File {path} does not exist in S3 bucket {self.config.bucket}") from e
                _raise_for_s3_error(e, f"generating presigned URL for {path}")

        session = get_http_session()

        async def _download() -> AsyncIterator[bytes]:
            try:
                async for chunk in download_url_streaming(
                    url,
                    session=session,
                    byte_range=byte_range,
                    chunk_size=self.config.read_chunk_size,
                ):
                    yield chunk
            except aiohttp.ClientResponseError as e:
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
        """Stream a file using boto3's managed PUT/multipart transfer."""
        if content_length is None:
            raise ValueError("content_length is required for S3 uploads")
        if content_length < 0:
            raise ValueError("content_length must be non-negative")

        def upload() -> None:
            # Serial classic transfers keep reads on this AnyIO worker and memory bounded.
            transfer = TransferConfig(use_threads=False, preferred_transfer_client="classic")
            transfer.multipart_chunksize = ChunksizeAdjuster().adjust_chunksize(
                transfer.multipart_chunksize, content_length
            )
            with (
                closing(boto3.Session().client("s3", **self._get_client_kwargs())) as client,
                BufferedReader(_UploadStream(fstream, content_length)) as source,
            ):
                client.upload_fileobj(source, self.config.bucket, self._full_key(path), Config=transfer)

        try:
            async with create_task_group() as group:
                group.start_soon(to_thread.run_sync, upload)
        except ExceptionGroup as errors:
            error = errors.exceptions[0]  # Only the upload worker can fail in this group.
            if isinstance(error, ClientError):
                _raise_for_s3_error(error, f"uploading {path}")
            raise error from None
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
        full_key = self._full_key(path)

        async with self._client() as client:
            try:
                head = await client.head_object(Bucket=self.config.bucket, Key=full_key)
                return FileInfo(path=path, size=head["ContentLength"])
            except ClientError as e:
                error_code = e.response.get("Error", {}).get("Code", "")
                if error_code in ("NoSuchKey", "NotFound", "404"):
                    raise NotFoundError(f"File {path} does not exist in S3 bucket {self.config.bucket}") from e
                _raise_for_s3_error(e, f"getting file info for {path}")

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

                logger.info(f"Deleted file {path} from S3 bucket {self.config.bucket}")
                return file_info
            except ClientError as e:
                error_code = e.response.get("Error", {}).get("Code", "")
                if error_code in ("NoSuchKey", "NotFound", "404"):
                    raise NotFoundError(f"File {path} does not exist in S3 bucket {self.config.bucket}") from e
                _raise_for_s3_error(e, f"deleting {path}")

    async def delete_all(self) -> None:
        """Delete all files owned by this fileset's prefix in the S3 bucket.

        S3 prefix matching is a raw string match, so listing with the configured
        prefix also returns neighbor keys (``foo-bar``, ``foo.txt``, ``foobar``).
        Only owned keys — the prefix object itself and everything under the
        directory boundary — are deleted; neighboring filesets are never
        touched. A failure after one page and before the next may leave some
        owned keys deleted; a later retry lists again and deletes the rest.
        """
        prefix = self.config.prefix or ""

        async with self._client() as client:
            try:
                # List with the raw prefix, then delete only owned keys
                paginator = client.get_paginator("list_objects_v2")
                async for page in paginator.paginate(
                    Bucket=self.config.bucket,
                    Prefix=prefix,
                ):
                    owned_keys = [obj["Key"] for obj in page.get("Contents", []) if self._owns_key(obj["Key"])]
                    if owned_keys:
                        delete_request: dict[str, Any] = {"Objects": [{"Key": key} for key in owned_keys]}
                        await client.delete_objects(
                            Bucket=self.config.bucket,
                            Delete=delete_request,
                        )

                logger.info(f"Deleted all files with prefix '{prefix}' from S3 bucket {self.config.bucket}")
            except ClientError as e:
                _raise_for_s3_error(e, f"deleting all files from bucket {self.config.bucket}")

    async def get_cache_path_key(self, path: str | None = None) -> str:
        """Return cache path that includes bucket and prefix for uniqueness.

        Format: cache/s3/{bucket}/{prefix}/{path}
        """
        prefix_part = f"/{self.config.prefix}" if self.config.prefix else ""
        base = f"cache/s3/{self.config.bucket}{prefix_part}"
        if path is None:
            return base
        return f"{base}/{path}"

    def get_duckdb_path(self, path: str) -> str:
        """Return S3 URI for DuckDB access.

        Format: s3://bucket/prefix/path
        """
        prefix = self.config.prefix.rstrip("/") if self.config.prefix else ""
        full_path = f"{prefix}/{path}" if prefix else path
        return f"s3://{self.config.bucket}/{full_path}"
