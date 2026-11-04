# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for S3 storage backend."""

from contextlib import asynccontextmanager
from typing import Any, AsyncIterator
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from botocore.exceptions import ClientError
from nhx.common.api.common import SecretRef
from nhx.common.files.storage_config import DEFAULT_READ_CHUNK_SIZE, S3StorageConfig
from nhx.core.files.app.backends.base import ByteRange
from nhx.core.files.app.backends.factory import storage_impl_factory
from nhx.core.files.app.backends.s3 import (
    S3AccessError,
    S3BackendError,
    S3ConfigError,
    S3StorageImpl,
    S3UnavailableError,
    _raise_for_s3_error,
)
from nhx.core.files.exceptions import NotFoundError


def _make_client_error(code: str, message: str = "Test error") -> ClientError:
    """Helper to create a ClientError with the given code."""
    return ClientError({"Error": {"Code": code, "Message": message}}, "TestOperation")


class AsyncIteratorMock:
    """Helper class to create an async iterator from a list."""

    def __init__(self, items):
        self.items = items
        self.index = 0

    def __aiter__(self):
        return self

    async def __anext__(self):
        if self.index >= len(self.items):
            raise StopAsyncIteration
        item = self.items[self.index]
        self.index += 1
        return item


@asynccontextmanager
async def mock_s3_client(impl: S3StorageImpl) -> AsyncIterator[AsyncMock]:
    """Context manager that patches the S3 client for an impl instance."""
    mock_client = AsyncMock()
    with patch.object(impl._session, "client") as mock_session_client:
        mock_session_client.return_value.__aenter__.return_value = mock_client
        yield mock_client


@pytest.fixture
def s3_impl() -> S3StorageImpl:
    """Default S3 impl for testing."""
    config = S3StorageConfig(
        bucket="test-bucket",
        prefix="test-prefix",
        region="us-east-1",
        use_sdk_auth=True,
    )
    return S3StorageImpl(config, {})


@pytest.fixture
def s3_impl_no_prefix() -> S3StorageImpl:
    """S3 impl without prefix for testing."""
    config = S3StorageConfig(bucket="test-bucket", region="us-east-1", use_sdk_auth=True)
    return S3StorageImpl(config, {})


class TestRaiseForS3Error:
    """Test error code mapping."""

    @pytest.mark.parametrize(
        ("error_code", "expected_exc", "expected_msg"),
        [
            ("AccessDenied", S3AccessError, "Access denied"),
            ("InvalidAccessKeyId", S3AccessError, "Access denied"),
            ("SignatureDoesNotMatch", S3AccessError, "Access denied"),
            ("403", S3AccessError, "Access denied"),
            ("NoSuchBucket", S3ConfigError, "Not found"),
            ("NoSuchKey", S3ConfigError, "Not found"),
            ("NotFound", S3ConfigError, "Not found"),
            ("404", S3ConfigError, "Not found"),
            ("ServiceUnavailable", S3UnavailableError, "S3 unavailable"),
            ("SlowDown", S3UnavailableError, "S3 unavailable"),
            ("Throttling", S3UnavailableError, "S3 unavailable"),
            ("RequestTimeout", S3UnavailableError, "S3 unavailable"),
            ("UnknownError", S3BackendError, "S3 error"),
        ],
    )
    def test_error_code_mapping(self, error_code, expected_exc, expected_msg):
        """Test that error codes are mapped to correct exceptions."""
        error = _make_client_error(error_code)
        with pytest.raises(expected_exc) as exc_info:
            _raise_for_s3_error(error)
        assert expected_msg in str(exc_info.value)

    def test_error_with_context(self):
        """Test that context is included in error message."""
        error = _make_client_error("AccessDenied")
        with pytest.raises(S3AccessError) as exc_info:
            _raise_for_s3_error(error, "downloading file.txt")
        assert "downloading file.txt" in str(exc_info.value)


class TestS3StorageImplInit:
    """Test S3StorageImpl initialization and configuration."""

    def test_init_with_credentials(self):
        """Test initialization with explicit credentials (use_sdk_auth=False)."""
        config = S3StorageConfig(
            bucket="test-bucket",
            prefix="test-prefix",
            region="us-east-1",
            endpoint_url="http://localhost:9000",
            use_sdk_auth=False,
            access_key_id_secret=SecretRef("access-key"),
            secret_access_key_secret=SecretRef("secret-key"),
        )
        secrets = {
            "access_key_id": "test-access-key",
            "secret_access_key": "test-secret-key",
        }
        impl = S3StorageImpl(config, secrets)

        kwargs = impl._get_client_kwargs()
        assert kwargs["region_name"] == "us-east-1"
        assert kwargs["endpoint_url"] == "http://localhost:9000"
        assert kwargs["aws_access_key_id"] == "test-access-key"
        assert kwargs["aws_secret_access_key"] == "test-secret-key"
        assert kwargs["config"].signature_version == "s3v4"

    def test_init_without_credentials(self):
        """Test initialization without credentials (uses SDK chain)."""
        config = S3StorageConfig(bucket="test-bucket", region="us-east-1", use_sdk_auth=True)
        impl = S3StorageImpl(config, {})

        kwargs = impl._get_client_kwargs()
        assert kwargs["region_name"] == "us-east-1"
        assert "aws_access_key_id" not in kwargs
        assert "aws_secret_access_key" not in kwargs

    def test_init_with_custom_signature_version(self):
        """Test initialization with custom signature version."""
        config = S3StorageConfig(
            bucket="test-bucket",
            region="us-east-1",
            signature_version="s3",
            use_sdk_auth=True,
        )
        impl = S3StorageImpl(config, {})
        assert impl._get_client_kwargs()["config"].signature_version == "s3"

    @pytest.mark.parametrize(
        ("prefix", "path", "expected"),
        [
            ("test-prefix", "file.txt", "test-prefix/file.txt"),
            ("", "file.txt", "file.txt"),
            ("deep/nested/prefix", "file.txt", "deep/nested/prefix/file.txt"),
        ],
    )
    def test_full_key(self, prefix, path, expected):
        """Test _full_key includes prefix correctly."""
        config = S3StorageConfig(bucket="test-bucket", prefix=prefix, use_sdk_auth=True)
        impl = S3StorageImpl(config, {})
        assert impl._full_key(path) == expected

    @pytest.mark.parametrize(
        ("prefix", "path", "expected"),
        [
            (
                "test-prefix",
                "models/model.bin",
                "cache/s3/test-bucket/test-prefix/models/model.bin",
            ),
            ("test-prefix", None, "cache/s3/test-bucket/test-prefix"),
            ("", "file.txt", "cache/s3/test-bucket/file.txt"),
        ],
    )
    async def test_get_cache_path_key(self, prefix, path, expected):
        """Test cache path key generation."""
        config = S3StorageConfig(bucket="test-bucket", prefix=prefix, use_sdk_auth=True)
        impl = S3StorageImpl(config, {})
        assert await impl.get_cache_path_key(path) == expected

    def test_config_owns_storage_data(self):
        """Deleting an S3 fileset removes objects under our prefix, so we own the data."""
        config = S3StorageConfig(bucket="test-bucket", prefix="p", use_sdk_auth=True)
        assert config.owns_storage_data is True


class TestS3StorageImplListFiles:
    """Test list_files method."""

    async def test_list_files_success(self, s3_impl):
        """Test listing files from S3 bucket."""
        mock_paginator = MagicMock()
        mock_paginator.paginate.return_value = AsyncIteratorMock(
            [
                {
                    "Contents": [
                        {"Key": "test-prefix/file1.txt", "Size": 100},
                        {"Key": "test-prefix/dir/file2.txt", "Size": 200},
                    ]
                }
            ]
        )

        async with mock_s3_client(s3_impl) as mock_client:
            mock_client.get_paginator = MagicMock(return_value=mock_paginator)
            files = await s3_impl.list_files()

        assert len(files) == 2
        assert files[0].path == "file1.txt"
        assert files[0].size == 100
        assert files[1].path == "dir/file2.txt"
        assert files[1].size == 200

    async def test_list_files_skips_directory_markers(self, s3_impl):
        """Test that directory markers (keys ending with /) are skipped."""
        mock_paginator = MagicMock()
        mock_paginator.paginate.return_value = AsyncIteratorMock(
            [
                {
                    "Contents": [
                        {"Key": "test-prefix/file.txt", "Size": 100},
                        {"Key": "test-prefix/dir/", "Size": 0},  # Directory marker
                    ]
                }
            ]
        )

        async with mock_s3_client(s3_impl) as mock_client:
            mock_client.get_paginator = MagicMock(return_value=mock_paginator)
            files = await s3_impl.list_files()

        assert len(files) == 1
        assert files[0].path == "file.txt"

    async def test_list_files_bucket_not_found(self, s3_impl_no_prefix):
        """Test list_files raises S3ConfigError when bucket not found."""
        mock_paginator = MagicMock()

        async def raise_error():
            raise _make_client_error("NoSuchBucket")
            yield  # noqa: B901 - unreachable but needed for generator

        mock_paginator.paginate.return_value = raise_error()

        async with mock_s3_client(s3_impl_no_prefix) as mock_client:
            mock_client.get_paginator = MagicMock(return_value=mock_paginator)
            with pytest.raises(S3ConfigError) as exc_info:
                await s3_impl_no_prefix.list_files()

        assert "Not found" in str(exc_info.value)


class TestS3StorageImplListFilesPathFilters:
    """Test list_files path filters with realistic S3 raw-prefix semantics.

    S3 treats Prefix as a raw string match, so the mocked paginator filters
    keys the same way: a key is returned when it starts with the query
    prefix. The fileset's own storage boundary (``foo/``) scopes the query,
    while the path filter keeps raw prefix matching within the fileset.
    """

    ALL_RELATIVE = [
        "item",
        "item.txt",
        "item/a.txt",
        "item.txt/a.txt",
        "item.txt.bak",
        "item-extra/a.txt",
    ]

    @staticmethod
    def _raw_prefix_paginator(keys: list[str]) -> MagicMock:
        """Paginator that mimics S3 raw string prefix matching."""
        captured: dict[str, Any] = {}

        def paginate(*, Bucket=None, Prefix=""):  # noqa: N803 - matches botocore API
            captured["Prefix"] = Prefix
            return AsyncIteratorMock([{"Contents": [{"Key": k, "Size": 1} for k in keys if k.startswith(Prefix)]}])

        paginator = MagicMock()
        paginator.paginate.side_effect = paginate
        paginator.captured = captured
        return paginator

    @pytest.mark.parametrize("prefix", ["foo", "foo/", ""])
    @pytest.mark.parametrize(
        ("path", "expected"),
        [
            (None, ALL_RELATIVE),
            ("", ALL_RELATIVE),
            ("item", ALL_RELATIVE),
            ("item/", ["item/a.txt"]),
            ("item.txt", ["item.txt", "item.txt/a.txt", "item.txt.bak"]),
            ("item.txt/", ["item.txt/a.txt"]),
        ],
    )
    async def test_list_files_path_filters(self, prefix, path, expected):
        """Path filters keep raw prefix matching inside the fileset boundary."""
        config = S3StorageConfig(bucket="test-bucket", prefix=prefix, region="us-east-1", use_sdk_auth=True)
        impl = S3StorageImpl(config, {})
        boundary = impl._full_key("")
        keys = [boundary + rel for rel in self.ALL_RELATIVE]

        paginator = self._raw_prefix_paginator(keys)
        async with mock_s3_client(impl) as mock_client:
            mock_client.get_paginator = MagicMock(return_value=paginator)
            files = await impl.list_files(path)

        assert sorted(f.path for f in files) == sorted(expected)
        # The query prefix is the fileset's storage boundary plus the raw path filter.
        assert paginator.captured["Prefix"] == boundary + (path or "")

    async def test_list_files_excludes_neighbor_keys(self):
        """Neighbor filesets and the bare prefix object are not listed."""
        config = S3StorageConfig(bucket="test-bucket", prefix="foo", region="us-east-1", use_sdk_auth=True)
        impl = S3StorageImpl(config, {})
        keys = [
            "foo",
            "foo/a.txt",
            "foo-bar/a.txt",
            "foo.txt",
            "foobar/a.txt",
        ]

        paginator = self._raw_prefix_paginator(keys)
        async with mock_s3_client(impl) as mock_client:
            mock_client.get_paginator = MagicMock(return_value=paginator)
            files = await impl.list_files()

        assert [f.path for f in files] == ["a.txt"]


class TestS3StorageImplDownload:
    """Test download method."""

    async def test_download_generates_presigned_url(self, s3_impl):
        """Test that download generates a presigned URL and streams content."""

        async def mock_download_streaming(url, **kwargs):
            assert "test-bucket" in url
            assert "X-Amz-Signature" in url
            yield b"hello world"

        async with mock_s3_client(s3_impl) as mock_client:
            mock_client.generate_presigned_url.return_value = (
                "https://test-bucket.s3.amazonaws.com/test-prefix/file.txt"
                "?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Signature=abc123"
            )
            with patch(
                "nhx.core.files.app.backends.s3.download_url_streaming",
                side_effect=mock_download_streaming,
            ):
                download_iter = await s3_impl.download("file.txt", None)
                chunks = [chunk async for chunk in download_iter]

        assert b"".join(chunks) == b"hello world"
        mock_client.generate_presigned_url.assert_called_once_with(
            "get_object",
            Params={"Bucket": "test-bucket", "Key": "test-prefix/file.txt"},
            ExpiresIn=3600,
        )

    async def test_download_passes_byte_range_and_chunk_size(self, s3_impl_no_prefix):
        """Test that download passes byte_range and chunk_size to streaming."""
        captured_kwargs = {}

        async def mock_download_streaming(url, **kwargs):
            captured_kwargs.update(kwargs)
            yield b"56789"

        byte_range = ByteRange(start=5, end=9)

        async with mock_s3_client(s3_impl_no_prefix) as mock_client:
            mock_client.generate_presigned_url.return_value = "https://test.com/file?signed=true"
            with patch(
                "nhx.core.files.app.backends.s3.download_url_streaming",
                side_effect=mock_download_streaming,
            ):
                download_iter = await s3_impl_no_prefix.download("file.txt", byte_range)
                chunks = [chunk async for chunk in download_iter]

        assert captured_kwargs["byte_range"] == byte_range
        assert captured_kwargs["chunk_size"] == DEFAULT_READ_CHUNK_SIZE
        assert b"".join(chunks) == b"56789"

    async def test_download_converts_404_to_not_found_error(self, s3_impl):
        """Test that HTTP 404 from presigned URL is converted to NotFoundError."""
        from unittest.mock import Mock

        import aiohttp
        from nhx.core.files.exceptions import NotFoundError

        async def mock_download_streaming_404(url, **kwargs):
            raise aiohttp.ClientResponseError(
                request_info=Mock(),
                history=(),
                status=404,
                message="Not Found",
            )
            yield  # pragma: no cover

        async with mock_s3_client(s3_impl) as mock_client:
            mock_client.generate_presigned_url.return_value = (
                "https://test-bucket.s3.amazonaws.com/test-prefix/missing.txt?X-Amz-Signature=abc123"
            )
            with patch(
                "nhx.core.files.app.backends.s3.download_url_streaming",
                side_effect=mock_download_streaming_404,
            ):
                download_iter = await s3_impl.download("missing.txt", None)
                with pytest.raises(NotFoundError) as exc_info:
                    async for _ in download_iter:
                        pass

        assert "missing.txt" in str(exc_info.value)
        assert "test-bucket" in str(exc_info.value)

    async def test_download_propagates_other_http_errors(self, s3_impl):
        """Test that non-404 HTTP errors are wrapped as S3BackendError."""
        from unittest.mock import Mock

        import aiohttp
        from nhx.core.files.app.backends.s3 import S3BackendError

        async def mock_download_streaming_500(url, **kwargs):
            raise aiohttp.ClientResponseError(
                request_info=Mock(),
                history=(),
                status=500,
                message="Internal Server Error",
            )
            yield  # pragma: no cover

        async with mock_s3_client(s3_impl) as mock_client:
            mock_client.generate_presigned_url.return_value = (
                "https://test-bucket.s3.amazonaws.com/test-prefix/file.txt?X-Amz-Signature=abc123"
            )
            with patch(
                "nhx.core.files.app.backends.s3.download_url_streaming",
                side_effect=mock_download_streaming_500,
            ):
                download_iter = await s3_impl.download("file.txt", None)
                with pytest.raises(S3BackendError) as exc_info:
                    async for _ in download_iter:
                        pass

        assert "500" in str(exc_info.value)


class TestS3StorageImplUpload:
    """Test upload method."""

    async def test_upload_requires_content_length(self, s3_impl_no_prefix):
        """Test upload raises ValueError when content_length is not provided."""

        async def data_stream():
            yield b"test data"

        with pytest.raises(ValueError) as exc_info:
            await s3_impl_no_prefix.upload("file.txt", data_stream())

        assert "content_length is required" in str(exc_info.value)

    async def test_upload_generates_presigned_url(self, s3_impl):
        """Test that upload generates a presigned URL and streams content."""
        captured_url = None
        captured_headers = None

        async def mock_upload_streaming(url, data, headers=None, **kwargs):
            nonlocal captured_url, captured_headers
            captured_url = url
            captured_headers = headers
            async for _ in data:
                pass
            return 0

        async def data_stream():
            yield b"test data"

        async with mock_s3_client(s3_impl) as mock_client:
            mock_client.generate_presigned_url.return_value = (
                "https://test-bucket.s3.amazonaws.com/test-prefix/file.txt"
                "?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Signature=abc123"
            )
            with patch(
                "nhx.core.files.app.backends.s3.upload_url_streaming",
                side_effect=mock_upload_streaming,
            ):
                result = await s3_impl.upload("file.txt", data_stream(), content_length=9)

        assert result.path == "file.txt"
        assert result.size == 9
        assert "test-bucket" in captured_url
        assert "X-Amz-Signature" in captured_url
        assert captured_headers["Content-Length"] == "9"

    async def test_upload_presigned_url_generation_fails(self, s3_impl_no_prefix):
        """Test upload handles presigned URL generation failure."""

        async def data_stream():
            yield b"test data"

        async with mock_s3_client(s3_impl_no_prefix) as mock_client:
            mock_client.generate_presigned_url.side_effect = _make_client_error("NoSuchBucket")
            with pytest.raises(S3ConfigError) as exc_info:
                await s3_impl_no_prefix.upload("file.txt", data_stream(), content_length=9)

        assert "Not found" in str(exc_info.value)


class TestS3StorageImplValidateStorage:
    """Test validate_storage method."""

    async def test_validate_storage_success(self, s3_impl_no_prefix):
        """Test successful storage validation."""
        async with mock_s3_client(s3_impl_no_prefix) as mock_client:
            await s3_impl_no_prefix.validate_storage()

        mock_client.head_bucket.assert_called_once_with(Bucket="test-bucket")

    async def test_validate_storage_bucket_not_found(self, s3_impl_no_prefix):
        """Test validation fails when bucket doesn't exist."""
        async with mock_s3_client(s3_impl_no_prefix) as mock_client:
            mock_client.head_bucket.side_effect = _make_client_error("404")
            with pytest.raises(S3ConfigError) as exc_info:
                await s3_impl_no_prefix.validate_storage()

        assert "Not found" in str(exc_info.value)


class TestS3StorageImplFileOperations:
    """Test get_file and delete methods."""

    async def test_get_file_success(self, s3_impl):
        """Test getting file metadata."""
        async with mock_s3_client(s3_impl) as mock_client:
            mock_client.head_object.return_value = {"ContentLength": 12}
            file_info = await s3_impl.get_file("file.txt")

        assert file_info.path == "file.txt"
        assert file_info.size == 12
        mock_client.head_object.assert_called_once_with(Bucket="test-bucket", Key="test-prefix/file.txt")

    async def test_get_file_not_found(self, s3_impl_no_prefix):
        """Test get_file raises NotFoundError when file doesn't exist."""
        async with mock_s3_client(s3_impl_no_prefix) as mock_client:
            mock_client.head_object.side_effect = _make_client_error("404")
            with pytest.raises(NotFoundError) as exc_info:
                await s3_impl_no_prefix.get_file("missing.txt")

        assert "missing.txt" in str(exc_info.value)

    async def test_delete_success(self, s3_impl):
        """Test successful file deletion."""
        async with mock_s3_client(s3_impl) as mock_client:
            mock_client.head_object.return_value = {"ContentLength": 9}
            result = await s3_impl.delete("file.txt")

        assert result.path == "file.txt"
        assert result.size == 9
        mock_client.delete_object.assert_called_once_with(Bucket="test-bucket", Key="test-prefix/file.txt")

    async def test_delete_not_found(self, s3_impl_no_prefix):
        """Test delete raises NotFoundError when file doesn't exist."""
        async with mock_s3_client(s3_impl_no_prefix) as mock_client:
            mock_client.head_object.side_effect = _make_client_error("404")
            with pytest.raises(NotFoundError) as exc_info:
                await s3_impl_no_prefix.delete("missing.txt")

        assert "missing.txt" in str(exc_info.value)


class TestS3StorageImplDeleteAll:
    """Test delete_all method."""

    async def test_delete_all_success(self, s3_impl):
        """Test successful bulk deletion."""
        mock_paginator = MagicMock()
        mock_paginator.paginate.return_value = AsyncIteratorMock(
            [
                {
                    "Contents": [
                        {"Key": "test-prefix/file1.txt"},
                        {"Key": "test-prefix/file2.txt"},
                    ]
                }
            ]
        )

        async with mock_s3_client(s3_impl) as mock_client:
            mock_client.get_paginator = MagicMock(return_value=mock_paginator)
            await s3_impl.delete_all()

        mock_client.delete_objects.assert_called_once()

    async def test_delete_all_empty(self, s3_impl):
        """Test delete_all on empty bucket doesn't call delete_objects."""
        mock_paginator = MagicMock()
        mock_paginator.paginate.return_value = AsyncIteratorMock([{"Contents": []}])

        async with mock_s3_client(s3_impl) as mock_client:
            mock_client.get_paginator = MagicMock(return_value=mock_paginator)
            await s3_impl.delete_all()

        mock_client.delete_objects.assert_not_called()


class TestS3StorageImplDeleteAllPrefixIsolation:
    """delete_all must only delete keys owned by this fileset's prefix.

    S3 prefix matching is a raw string match, so listing with Prefix=foo also
    returns foo-bar, foo.txt, and foobar. Those neighbor keys must never be
    deleted: a key is owned only when it equals the configured prefix or lies
    under the directory boundary (``foo/``).
    """

    @staticmethod
    def _deleted_keys(mock_client) -> list[str]:
        """Collect every key passed to delete_objects across all calls."""
        return [
            obj["Key"] for call in mock_client.delete_objects.call_args_list for obj in call.kwargs["Delete"]["Objects"]
        ]

    @staticmethod
    def _make_page(keys: list[str]) -> dict[str, list[dict[str, Any]]]:
        return {"Contents": [{"Key": k} for k in keys]}

    @staticmethod
    def _make_impl(prefix: str) -> S3StorageImpl:
        config = S3StorageConfig(bucket="test-bucket", prefix=prefix, region="us-east-1", use_sdk_auth=True)
        return S3StorageImpl(config, {})

    @staticmethod
    def _owned_and_neighbor_keys(prefix: str) -> tuple[set[str], list[str]]:
        """Owned keys and raw-prefix neighbor keys for a configured prefix."""
        bare = prefix.rstrip("/")
        boundary = bare + "/"
        owned = {prefix, boundary, boundary + "a.txt", boundary + "report.txt/a.txt"}
        if prefix.endswith("/"):
            # A slash-terminated prefix owns its directory tree; only the bare
            # prefix object (outside the directory) is a neighbor.
            neighbors = [bare]
        else:
            neighbors = [prefix + suffix for suffix in ("-bar", "-bar/a.txt", ".txt", ".txt/a.txt", "bar", "bar/a.txt")]
        return owned, neighbors

    @pytest.mark.parametrize("prefix", ["foo", "foo.txt", "foo/", "foo.txt/"])
    async def test_delete_all_keeps_neighbor_keys(self, prefix):
        """Only the exact prefix object and directory descendants are deleted."""
        impl = self._make_impl(prefix)
        owned, neighbors = self._owned_and_neighbor_keys(prefix)

        mock_paginator = MagicMock()
        mock_paginator.paginate.return_value = AsyncIteratorMock([self._make_page(sorted(owned) + neighbors)])

        async with mock_s3_client(impl) as mock_client:
            mock_client.get_paginator = MagicMock(return_value=mock_paginator)
            await impl.delete_all()

        assert set(self._deleted_keys(mock_client)) == owned
        mock_paginator.paginate.assert_called_once_with(Bucket="test-bucket", Prefix=prefix)

    @pytest.mark.parametrize("prefix", ["foo", "foo.txt"])
    async def test_delete_all_skips_page_of_only_neighbors(self, prefix):
        """A page holding only neighbor keys must not call delete_objects."""
        impl = self._make_impl(prefix)
        owned, neighbors = self._owned_and_neighbor_keys(prefix)

        mock_paginator = MagicMock()
        mock_paginator.paginate.return_value = AsyncIteratorMock(
            [
                self._make_page(neighbors),
                self._make_page(sorted(owned)),
            ]
        )

        async with mock_s3_client(impl) as mock_client:
            mock_client.get_paginator = MagicMock(return_value=mock_paginator)
            await impl.delete_all()

        assert mock_client.delete_objects.call_count == 1
        assert set(self._deleted_keys(mock_client)) == owned

    @pytest.mark.parametrize("prefix", ["foo", "foo.txt"])
    async def test_delete_all_one_call_per_owned_page(self, prefix):
        """One delete_objects call per page that still has owned keys."""
        impl = self._make_impl(prefix)
        owned, neighbors = self._owned_and_neighbor_keys(prefix)

        mock_paginator = MagicMock()
        mock_paginator.paginate.return_value = AsyncIteratorMock(
            [
                self._make_page(sorted(owned)),
                {"Contents": []},
                self._make_page(neighbors + [next(iter(owned))]),
            ]
        )

        async with mock_s3_client(impl) as mock_client:
            mock_client.get_paginator = MagicMock(return_value=mock_paginator)
            await impl.delete_all()

        assert mock_client.delete_objects.call_count == 2
        assert set(self._deleted_keys(mock_client)) == owned

    async def test_delete_all_empty_prefix_deletes_everything(self):
        """An empty prefix owns the whole bucket, so every listed key is deleted."""
        impl = self._make_impl("")
        keys = ["a.txt", "dir/b.txt", "foo-bar", "foo.txt"]

        mock_paginator = MagicMock()
        mock_paginator.paginate.return_value = AsyncIteratorMock([self._make_page(keys)])

        async with mock_s3_client(impl) as mock_client:
            mock_client.get_paginator = MagicMock(return_value=mock_paginator)
            await impl.delete_all()

        assert set(self._deleted_keys(mock_client)) == set(keys)


class TestAiobotocoreCompatibility:
    """Regression tests for aiobotocore/botocore version compatibility.

    All other tests mock impl._session.client(), which bypasses aiobotocore
    entirely. These tests let aiobotocore create a real client to catch cases
    where botocore was upgraded beyond what aiobotocore supports.

    See: pyproject.toml botocore version constraint.
    """

    async def test_client_creation_does_not_raise_type_error(self):
        """Regression: botocore must be within aiobotocore's supported version range.

        When botocore is too new, aiobotocore fails during client creation with:
            TypeError: compute_endpoint_resolver_builtin_defaults() missing 1
            required positional argument: 's3_disable_express_session_auth'
        """
        config = S3StorageConfig(
            bucket="test-bucket",
            region="us-east-1",
            use_sdk_auth=False,
            access_key_id_secret=SecretRef("access_key_id"),
            secret_access_key_secret=SecretRef("secret_access_key"),
        )
        impl = S3StorageImpl(config, {"access_key_id": "test-key", "secret_access_key": "test-secret"})

        try:
            async with impl._client():
                pass
        except TypeError as e:
            pytest.fail(
                f"aiobotocore/botocore version incompatibility: {e}. "
                "The botocore upper bound in pyproject.toml likely needs tightening "
                "to match what aiobotocore (via aioboto3) supports."
            )
        except Exception:
            pass  # Network errors, auth errors, etc. are fine — client was created successfully


class TestS3StorageFactory:
    """Test factory creates S3StorageImpl correctly."""

    def test_factory_creates_s3_impl(self):
        """Test that factory correctly creates S3 storage impl."""
        config = S3StorageConfig(bucket="test-bucket", region="us-east-1", use_sdk_auth=True)
        impl = storage_impl_factory(config, {})

        assert isinstance(impl, S3StorageImpl)
        assert impl.config == config
