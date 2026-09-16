# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Integration tests for S3 storage backend.

These tests connect to an actual S3-compatible storage instance and make
real HTTP requests through the Files service SDK. They are skipped by default
(see conftest.py in this directory).

To run these tests with RustFS:

    # Start RustFS
    docker run --rm --name rustfs_local -p 9000:9000 -p 9001:9001 \
        -v rustfs_data:/data \
        -e RUSTFS_ACCESS_KEY=rustfsadmin -e RUSTFS_SECRET_KEY=rustfsadmin \
        rustfs/rustfs:latest /data

    # Run tests
    RUN_EXTERNAL_STORAGE_TESTS=1 uv run --frozen pytest \
        services/core/files/tests/integration/external_storage/test_s3_storage.py -v

Environment variables for custom S3 endpoints:
    S3_TEST_ENDPOINT    - S3 endpoint URL (default: http://localhost:9000)
    S3_TEST_ACCESS_KEY  - Access key (default: rustfsadmin)
    S3_TEST_SECRET_KEY  - Secret key (default: rustfsadmin)
    S3_TEST_REGION      - AWS region (default: us-east-1)
"""

from __future__ import annotations

import os
import uuid
from collections.abc import AsyncIterator, Iterator

import pytest
from aiobotocore.session import get_session
from botocore.exceptions import ClientError
from nemo_helix import NeMoHelix
from nemo_helix_plugin.client.adapter import client_from_platform
from nemo_helix_plugin.client.errors import NemoHTTPError as ClientBadRequestError
from nemo_helix_plugin.files.client import FilesClient
from nemo_helix_plugin.files.types import CreateFilesetRequest, FilesetOutput
from nemo_helix_plugin.secrets.client import SecretsClient
from nemo_helix_plugin.secrets.types import PlatformSecretCreateRequest
from nhx.common.auth import AuthClient, get_auth_client
from nhx.common.auth.models import Principal
from nhx.common.config import AuthConfig
from nhx.common.files.storage_config import S3StorageConfig
from nhx.core.files.config import FilesConfig
from nhx.core.files.service import FilesService
from nhx.core.files.testing.utils import create_fileset
from nhx.core.secrets.service import SecretsService
from nhx.testing import create_test_client
from pydantic import SecretStr
from types_aiobotocore_s3 import S3Client

# Test configuration - uses RustFS by default
S3_TEST_ENDPOINT = os.environ.get("S3_TEST_ENDPOINT", "http://localhost:9000")
S3_TEST_ACCESS_KEY = os.environ.get("S3_TEST_ACCESS_KEY", "rustfsadmin")
S3_TEST_SECRET_KEY = os.environ.get("S3_TEST_SECRET_KEY", "rustfsadmin")
S3_TEST_REGION = os.environ.get("S3_TEST_REGION", "us-east-1")
DEFAULT_WORKSPACE = "default"


@pytest.fixture
async def s3_client() -> AsyncIterator[S3Client]:
    """Create an S3 client connected to the test endpoint."""
    session = get_session()
    async with session.create_client(
        "s3",
        endpoint_url=S3_TEST_ENDPOINT,
        region_name=S3_TEST_REGION,
        aws_access_key_id=S3_TEST_ACCESS_KEY,
        aws_secret_access_key=S3_TEST_SECRET_KEY,
    ) as client:
        yield client


def s3_storage_config(
    bucket: str,
    access_key_secret: str,
    secret_key_secret: str,
    prefix: str | None = None,
) -> dict[str, object]:
    """Helper to create S3 storage config dict with explicit credentials."""
    config: dict[str, object] = {
        "type": "s3",
        "bucket": bucket,
        "endpoint_url": S3_TEST_ENDPOINT,
        "region": S3_TEST_REGION,
        "use_sdk_auth": False,
        "access_key_id_secret": access_key_secret,
        "secret_access_key_secret": secret_key_secret,
    }
    if prefix:
        config["prefix"] = prefix
    return config


@pytest.fixture
async def s3_test_bucket(s3_client: S3Client) -> AsyncIterator[str]:
    """Create a unique test bucket for each test, then clean up."""
    bucket_name = f"test-bucket-{uuid.uuid4().hex[:8]}"

    await s3_client.create_bucket(Bucket=bucket_name)
    yield bucket_name

    # Cleanup: delete all objects and the bucket
    try:
        paginator = s3_client.get_paginator("list_objects_v2")
        async for page in paginator.paginate(Bucket=bucket_name):
            objects = page.get("Contents", [])
            if objects:
                await s3_client.delete_objects(
                    Bucket=bucket_name,
                    Delete={"Objects": [{"Key": obj["Key"]} for obj in objects]},
                )
        await s3_client.delete_bucket(Bucket=bucket_name)
    except ClientError:
        pass


@pytest.fixture
def s3_credentials(sdk: NeMoHelix) -> Iterator[tuple[str, str]]:
    """Create temporary secrets for S3 credentials and clean up after use."""
    access_key_secret = f"s3-access-key-{uuid.uuid4().hex[:8]}"
    secret_key_secret = f"s3-secret-key-{uuid.uuid4().hex[:8]}"

    secrets = client_from_platform(sdk, SecretsClient)
    secrets.create_secret(
        body=PlatformSecretCreateRequest(name=access_key_secret, value=SecretStr(S3_TEST_ACCESS_KEY)),
        workspace=DEFAULT_WORKSPACE,
    )
    secrets.create_secret(
        body=PlatformSecretCreateRequest(name=secret_key_secret, value=SecretStr(S3_TEST_SECRET_KEY)),
        workspace=DEFAULT_WORKSPACE,
    )

    try:
        yield (access_key_secret, secret_key_secret)
    finally:
        secrets.delete_secret(name=access_key_secret, workspace=DEFAULT_WORKSPACE)
        secrets.delete_secret(name=secret_key_secret, workspace=DEFAULT_WORKSPACE)


@pytest.fixture
def s3_fileset(sdk: NeMoHelix, s3_test_bucket: str, s3_credentials: tuple[str, str]) -> Iterator[FilesetOutput]:
    """Create a fileset with S3 storage for testing."""
    name = f"s3-test-{uuid.uuid4().hex[:8]}"
    access_key_secret, secret_key_secret = s3_credentials

    with create_fileset(
        sdk,
        name,
        storage=s3_storage_config(s3_test_bucket, access_key_secret, secret_key_secret),
    ) as fileset:
        yield fileset


class TestS3StorageBackend:
    """Test S3 storage backend through the Files service SDK."""

    def test_fileset_create_with_s3_storage(self, sdk: NeMoHelix, s3_test_bucket: str, s3_credentials: tuple[str, str]):
        """Test creating a fileset with S3 storage configuration."""
        name = f"s3-test-{uuid.uuid4().hex[:8]}"
        access_key_secret, secret_key_secret = s3_credentials

        with create_fileset(
            sdk,
            name,
            storage=s3_storage_config(s3_test_bucket, access_key_secret, secret_key_secret),
        ) as fileset:
            files = client_from_platform(sdk, FilesClient)
            persisted = files.get_fileset(name=fileset.name, workspace=fileset.workspace).data()
            assert persisted.storage.type == "s3"
            assert persisted.storage.bucket == s3_test_bucket

    def test_validate_storage_bucket_not_found(self, sdk: NeMoHelix, s3_credentials: tuple[str, str]):
        """Test that creating a fileset with non-existent bucket fails validation."""
        name = f"s3-test-{uuid.uuid4().hex[:8]}"
        access_key_secret, secret_key_secret = s3_credentials

        files = client_from_platform(sdk, FilesClient)
        with pytest.raises(ClientBadRequestError) as exc_info:
            files.create_fileset(
                workspace=DEFAULT_WORKSPACE,
                body=CreateFilesetRequest(
                    name=name,
                    storage=s3_storage_config(
                        f"nonexistent-bucket-{uuid.uuid4().hex[:8]}",
                        access_key_secret,
                        secret_key_secret,
                    ),
                ),
            )

        assert exc_info.value.status_code == 400
        assert "Not found" in str(exc_info.value) or "bucket" in str(exc_info.value).lower()

    def test_invalid_credentials(self, sdk: NeMoHelix, s3_test_bucket: str):
        """Test that invalid credentials raise an error during fileset creation."""
        name = f"s3-test-{uuid.uuid4().hex[:8]}"
        bad_access_secret = f"bad-s3-access-{uuid.uuid4().hex[:8]}"
        bad_secret_secret = f"bad-s3-secret-{uuid.uuid4().hex[:8]}"

        secrets = client_from_platform(sdk, SecretsClient)
        secrets.create_secret(
            body=PlatformSecretCreateRequest(name=bad_access_secret, value=SecretStr("invalid-key")),
            workspace=DEFAULT_WORKSPACE,
        )
        secrets.create_secret(
            body=PlatformSecretCreateRequest(name=bad_secret_secret, value=SecretStr("invalid-secret")),
            workspace=DEFAULT_WORKSPACE,
        )

        try:
            files = client_from_platform(sdk, FilesClient)
            with pytest.raises(ClientBadRequestError) as exc_info:
                files.create_fileset(
                    workspace=DEFAULT_WORKSPACE,
                    body=CreateFilesetRequest(
                        name=name,
                        storage=s3_storage_config(s3_test_bucket, bad_access_secret, bad_secret_secret),
                    ),
                )

            assert exc_info.value.status_code == 400
            assert "Access denied" in str(exc_info.value) or "credentials" in str(exc_info.value).lower()
        finally:
            secrets.delete_secret(name=bad_access_secret, workspace=DEFAULT_WORKSPACE)
            secrets.delete_secret(name=bad_secret_secret, workspace=DEFAULT_WORKSPACE)

    def test_upload_and_download_roundtrip(self, sdk: NeMoHelix, s3_fileset: FilesetOutput, tmp_path):
        """Test upload file, download it back, verify content matches."""
        test_content = b"Hello, S3 storage backend test!"
        upload_file = tmp_path / "test-file.txt"
        upload_file.write_bytes(test_content)

        files = client_from_platform(sdk, FilesClient)
        files.upload_file(
            content=test_content,
            path="test-file.txt",
            name=s3_fileset.name,
            workspace=s3_fileset.workspace,
        )

        # Verify via list
        listing = files.list_files(name=s3_fileset.name, workspace=s3_fileset.workspace).data().data
        assert len(listing) == 1
        assert listing[0].path == "test-file.txt"
        assert listing[0].size == len(test_content)

        # Download to memory and verify
        downloaded_content = files.download_file(
            name=s3_fileset.name, workspace=s3_fileset.workspace, path="test-file.txt"
        ).read()
        assert downloaded_content == test_content

        # Download to a file and verify
        download_path = tmp_path / "downloaded.txt"
        download_path.write_bytes(
            files.download_file(name=s3_fileset.name, workspace=s3_fileset.workspace, path="test-file.txt").read()
        )
        assert download_path.read_bytes() == test_content

    def test_upload_and_download_empty_file(self, sdk: NeMoHelix, s3_fileset: FilesetOutput, tmp_path):
        """Test upload and download of an empty file.

        This exercises the edge case where iter_chunked yields no chunks,
        testing that the preflight logic handles StopAsyncIteration correctly.
        """
        test_content = b""
        upload_file = tmp_path / "empty-file.txt"
        upload_file.write_bytes(test_content)

        files = client_from_platform(sdk, FilesClient)
        files.upload_file(
            content=upload_file.read_bytes(),
            path="empty-file.txt",
            name=s3_fileset.name,
            workspace=s3_fileset.workspace,
        )

        # Verify via list
        listing = files.list_files(name=s3_fileset.name, workspace=s3_fileset.workspace).data().data
        assert len(listing) == 1
        assert listing[0].path == "empty-file.txt"
        assert listing[0].size == 0

        # Download to memory and verify
        downloaded_content = files.download_file(
            name=s3_fileset.name, workspace=s3_fileset.workspace, path="empty-file.txt"
        ).read()
        assert downloaded_content == test_content

    def test_upload_large_file(self, sdk: NeMoHelix, s3_fileset: FilesetOutput, tmp_path):
        """Test upload of a large file via presigned URL streaming.

        Validates that large file uploads work correctly through the presigned
        PUT URL with aiohttp streaming. Uses a 6MB file to exercise the streaming
        path with a non-trivial payload.
        """
        test_content = b"x" * (6 * 1024 * 1024)  # 6MB
        upload_file = tmp_path / "large-file.bin"
        upload_file.write_bytes(test_content)

        files = client_from_platform(sdk, FilesClient)
        files.upload_file(
            content=upload_file.read_bytes(),
            path="large-file.bin",
            name=s3_fileset.name,
            workspace=s3_fileset.workspace,
        )

        listing = files.list_files(name=s3_fileset.name, workspace=s3_fileset.workspace).data().data
        assert len(listing) == 1
        assert listing[0].size == len(test_content)

        downloaded_content = files.download_file(
            name=s3_fileset.name, workspace=s3_fileset.workspace, path="large-file.bin"
        ).read()
        assert downloaded_content == test_content

    def test_download_with_byte_range(self, sdk: NeMoHelix, s3_fileset: FilesetOutput, tmp_path):
        """Test partial download using HTTP Range header."""
        test_content = b"0123456789ABCDEF"
        upload_file = tmp_path / "range-test.txt"
        upload_file.write_bytes(test_content)

        client_from_platform(sdk, FilesClient).upload_file(
            content=upload_file.read_bytes(),
            path="range-test.txt",
            name=s3_fileset.name,
            workspace=s3_fileset.workspace,
        )

        # Download bytes 5-10 (inclusive) using range header
        range_response = (
            client_from_platform(sdk, FilesClient)
            .with_headers({"Range": "bytes=5-10"})
            .download_file(name=s3_fileset.name, workspace=s3_fileset.workspace, path="range-test.txt")
        )
        range_content = range_response.read()
        assert range_response.http_response.status_code == 206  # Partial Content
        assert range_content == b"56789A"

    def test_delete_file(self, sdk: NeMoHelix, s3_fileset: FilesetOutput, tmp_path):
        """Test upload, delete, verify gone."""
        upload_file = tmp_path / "to-delete.txt"
        upload_file.write_bytes(b"Delete me!")

        files = client_from_platform(sdk, FilesClient)
        files.upload_file(
            content=upload_file.read_bytes(),
            path="to-delete.txt",
            name=s3_fileset.name,
            workspace=s3_fileset.workspace,
        )

        assert len(files.list_files(name=s3_fileset.name, workspace=s3_fileset.workspace).data().data) == 1

        files.delete_file(
            path="to-delete.txt",
            name=s3_fileset.name,
            workspace=s3_fileset.workspace,
        )

        assert len(files.list_files(name=s3_fileset.name, workspace=s3_fileset.workspace).data().data) == 0

    def test_delete_fileset_with_files(
        self,
        sdk: NeMoHelix,
        s3_test_bucket: str,
        s3_credentials: tuple[str, str],
        s3_client: S3Client,
        tmp_path,
    ):
        """Test deleting a fileset removes all files from S3.

        This test exercises the delete_all() method in the S3 backend which
        uses the DeleteObjects API. Some S3-compatible backends (like Oracle
        Object Storage) require specific headers for this operation.
        """
        name = f"s3-delete-test-{uuid.uuid4().hex[:8]}"
        access_key_secret, secret_key_secret = s3_credentials
        prefix = f"delete-test-{uuid.uuid4().hex[:8]}"

        # Create fileset with a unique prefix so we can verify cleanup
        files_client = client_from_platform(sdk, FilesClient)
        fileset = files_client.create_fileset(
            workspace=DEFAULT_WORKSPACE,
            body=CreateFilesetRequest(
                name=name,
                storage=s3_storage_config(s3_test_bucket, access_key_secret, secret_key_secret, prefix=prefix),
            ),
        ).data()

        try:
            # Upload multiple files to exercise bulk delete
            files_to_upload = {
                "file1.txt": b"content1",
                "file2.txt": b"content2",
                "subdir/file3.txt": b"content3",
            }

            for remote_path, content in files_to_upload.items():
                files_client.upload_file(
                    content=content,
                    path=remote_path,
                    name=fileset.name,
                    workspace=fileset.workspace,
                )

            # Verify files exist
            file_list = (
                client_from_platform(sdk, FilesClient).list_files(name=fileset.name, workspace=fileset.workspace).data()
            )
            assert len(file_list.data) == 3

            # Delete the fileset - this calls delete_all() on the S3 backend
            files_client.delete_fileset(name=fileset.name, workspace=fileset.workspace)

            # Verify fileset is gone
            from nemo_helix_plugin.client.errors import NotFoundError as ClientNotFoundError

            with pytest.raises(ClientNotFoundError):
                files_client.get_fileset(name=name, workspace=DEFAULT_WORKSPACE)

        except Exception:
            # Cleanup on failure - delete fileset if it still exists
            try:
                files_client.delete_fileset(name=name, workspace=DEFAULT_WORKSPACE)
            except Exception:
                pass
            raise

    def test_multiple_files_with_directory_structure(self, sdk: NeMoHelix, s3_fileset: FilesetOutput, tmp_path):
        """Test uploading multiple files with directory structure."""
        files_to_upload = {
            "file1.txt": b"content1",
            "dir/file2.txt": b"content2",
            "dir/subdir/file3.txt": b"content3",
        }

        for remote_path, content in files_to_upload.items():
            client_from_platform(sdk, FilesClient).upload_file(
                content=content,
                path=remote_path,
                name=s3_fileset.name,
                workspace=s3_fileset.workspace,
            )

        files = (
            client_from_platform(sdk, FilesClient)
            .list_files(name=s3_fileset.name, workspace=s3_fileset.workspace)
            .data()
            .data
        )
        paths = {f.path for f in files}
        assert paths == set(files_to_upload.keys())

    def test_prefix_isolation(
        self,
        sdk: NeMoHelix,
        s3_test_bucket: str,
        s3_credentials: tuple[str, str],
        tmp_path,
    ):
        """Test that filesets with different prefixes are isolated."""
        access_key_secret, secret_key_secret = s3_credentials
        prefix1 = f"prefix1-{uuid.uuid4().hex[:8]}"
        prefix2 = f"prefix2-{uuid.uuid4().hex[:8]}"

        with create_fileset(
            sdk,
            f"s3-test-{uuid.uuid4().hex[:8]}",
            storage=s3_storage_config(s3_test_bucket, access_key_secret, secret_key_secret, prefix=prefix1),
        ) as fileset1:
            with create_fileset(
                sdk,
                f"s3-test-{uuid.uuid4().hex[:8]}",
                storage=s3_storage_config(s3_test_bucket, access_key_secret, secret_key_secret, prefix=prefix2),
            ) as fileset2:
                # Upload to each fileset
                files = client_from_platform(sdk, FilesClient)
                files.upload_file(
                    content=b"fileset1 content",
                    path="file.txt",
                    name=fileset1.name,
                    workspace=fileset1.workspace,
                )

                files.upload_file(
                    content=b"fileset2 content",
                    path="file.txt",
                    name=fileset2.name,
                    workspace=fileset2.workspace,
                )

                # Verify isolation
                files1 = (
                    client_from_platform(sdk, FilesClient)
                    .list_files(name=fileset1.name, workspace=fileset1.workspace)
                    .data()
                )
                assert len(files1.data) == 1

                files2 = (
                    client_from_platform(sdk, FilesClient)
                    .list_files(name=fileset2.name, workspace=fileset2.workspace)
                    .data()
                )
                assert len(files2.data) == 1

                # Verify content isolation
                content1 = (
                    client_from_platform(sdk, FilesClient)
                    .download_file(name=fileset1.name, workspace=fileset1.workspace, path="file.txt")
                    .read()
                )
                content2 = (
                    client_from_platform(sdk, FilesClient)
                    .download_file(name=fileset2.name, workspace=fileset2.workspace, path="file.txt")
                    .read()
                )
                assert content1 == b"fileset1 content"
                assert content2 == b"fileset2 content"


@pytest.fixture
def sdk_with_s3_default(s3_test_bucket: str, monkeypatch: pytest.MonkeyPatch) -> Iterator[NeMoHelix]:
    """Create an SDK with S3 as the default storage config."""
    # Set AWS credentials via environment variables (SDK credential chain)
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", S3_TEST_ACCESS_KEY)
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", S3_TEST_SECRET_KEY)

    files_config = FilesConfig(
        default_storage_config=S3StorageConfig(
            bucket=s3_test_bucket,
            prefix="default-storage",
            region=S3_TEST_REGION,
            endpoint_url=S3_TEST_ENDPOINT,
            use_sdk_auth=True,
        )
    )  # type: ignore[abstract]

    mock_auth = AuthClient(
        principal=Principal(id="test@example.com"),
        config=AuthConfig(enabled=False),  # type: ignore[abstract]
    )

    with create_test_client(
        FilesService,
        SecretsService,
        service_configs={FilesService: files_config},
        dependency_overrides={get_auth_client: lambda: mock_auth},
    ) as sdk:
        yield sdk


class TestS3DefaultStorageConfig:
    """Test S3 as the default storage config for the Files service.

    These tests verify that when S3 is configured as the default storage backend,
    filesets created without explicit storage configuration inherit S3 storage.
    This simulates a deployment where S3 is the primary storage for all filesets.
    """

    def test_fileset_without_storage_uses_s3_default(self, sdk_with_s3_default: NeMoHelix, s3_test_bucket: str):
        """Test that filesets created without storage config use S3 default."""
        name = f"default-storage-test-{uuid.uuid4().hex[:8]}"

        # Create fileset WITHOUT specifying storage - should use S3 default
        files = client_from_platform(sdk_with_s3_default, FilesClient)
        fileset = files.create_fileset(
            workspace=DEFAULT_WORKSPACE,
            body=CreateFilesetRequest(name=name),
        ).data()

        try:
            # Verify it was created with S3 storage
            assert fileset.storage.type == "s3"
            assert fileset.storage.bucket == s3_test_bucket
            # Prefix should include the fileset path
            assert "default-storage" in fileset.storage.prefix
            assert name in fileset.storage.prefix
        finally:
            files.delete_fileset(name=name, workspace=DEFAULT_WORKSPACE)

    def test_upload_download_with_s3_default(self, sdk_with_s3_default: NeMoHelix, tmp_path):
        """Test file upload/download on fileset using S3 default storage."""
        name = f"default-storage-test-{uuid.uuid4().hex[:8]}"
        test_content = b"Hello from S3 default storage!"

        # Create fileset without explicit storage
        files = client_from_platform(sdk_with_s3_default, FilesClient)
        fileset = files.create_fileset(
            workspace=DEFAULT_WORKSPACE,
            body=CreateFilesetRequest(name=name),
        ).data()

        try:
            # Upload file
            upload_file = tmp_path / "test.txt"
            upload_file.write_bytes(test_content)

            files.upload_file(
                content=upload_file.read_bytes(),
                path="test.txt",
                name=fileset.name,
                workspace=fileset.workspace,
            )

            # List files
            files_response = sdk_with_s3_default.files.list(
                fileset=fileset.name,
                workspace=fileset.workspace,
            )
            assert len(files_response) == 1
            assert files_response[0].path == "test.txt"

            # Download and verify content
            downloaded = sdk_with_s3_default.files.download_content(
                remote_path="test.txt",
                fileset=fileset.name,
                workspace=fileset.workspace,
            )
            assert downloaded == test_content
        finally:
            files.delete_fileset(name=name, workspace=DEFAULT_WORKSPACE)

    def test_multiple_filesets_isolated_with_s3_default(self, sdk_with_s3_default: NeMoHelix, tmp_path):
        """Test that multiple filesets using S3 default are isolated via prefix."""
        name1 = f"default-test-1-{uuid.uuid4().hex[:8]}"
        name2 = f"default-test-2-{uuid.uuid4().hex[:8]}"

        files = client_from_platform(sdk_with_s3_default, FilesClient)
        fileset1 = files.create_fileset(workspace=DEFAULT_WORKSPACE, body=CreateFilesetRequest(name=name1)).data()
        fileset2 = files.create_fileset(workspace=DEFAULT_WORKSPACE, body=CreateFilesetRequest(name=name2)).data()

        try:
            # Both should have different prefixes (S3 storage has prefix attribute)
            storage1 = fileset1.storage
            storage2 = fileset2.storage
            assert storage1.type == "s3" and storage2.type == "s3"
            assert storage1.prefix != storage2.prefix
            assert name1 in storage1.prefix
            assert name2 in storage2.prefix

            # Upload same filename to each
            file1 = tmp_path / "file1.txt"
            file1.write_bytes(b"content for fileset 1")
            files.upload_file(
                content=file1.read_bytes(),
                path="shared-name.txt",
                name=fileset1.name,
                workspace=fileset1.workspace,
            )

            file2 = tmp_path / "file2.txt"
            file2.write_bytes(b"content for fileset 2")
            files.upload_file(
                content=file2.read_bytes(),
                path="shared-name.txt",
                name=fileset2.name,
                workspace=fileset2.workspace,
            )

            # Verify isolation
            content1 = sdk_with_s3_default.files.download_content(
                remote_path="shared-name.txt",
                fileset=fileset1.name,
                workspace=fileset1.workspace,
            )
            content2 = sdk_with_s3_default.files.download_content(
                remote_path="shared-name.txt",
                fileset=fileset2.name,
                workspace=fileset2.workspace,
            )

            assert content1 == b"content for fileset 1"
            assert content2 == b"content for fileset 2"
        finally:
            files.delete_fileset(name=name1, workspace=DEFAULT_WORKSPACE)
            files.delete_fileset(name=name2, workspace=DEFAULT_WORKSPACE)

    def test_download_from_huggingface_fileset_with_s3_default(self, sdk_with_s3_default: NeMoHelix):
        """Test downloading from a HuggingFace fileset works with S3 as default storage.

        This exercises the full download path with HuggingFace backend while the
        Files service is configured with S3 as default. The download uses the
        HuggingFace backend's streaming with preflight validation, ensuring that
        connection errors surface immediately (not after response headers are committed).
        """
        name = f"hf-with-s3-default-{uuid.uuid4().hex[:8]}"

        # Create a HuggingFace-backed fileset (explicitly specifying storage)
        files_client = client_from_platform(sdk_with_s3_default, FilesClient)
        fileset = files_client.create_fileset(
            workspace=DEFAULT_WORKSPACE,
            body=CreateFilesetRequest(
                name=name,
                storage={
                    "type": "huggingface",
                    "repo_id": "hf-internal-testing/tiny-random-bert",
                    "repo_type": "model",
                    "revision": "main",
                },
            ),
        ).data()

        try:
            # List files to verify connection works
            files = sdk_with_s3_default.files.list(
                fileset=fileset.name,
                workspace=fileset.workspace,
            )
            assert len(files) > 0, "Expected files in the HuggingFace repo"

            # Find config.json (typically small and always present in model repos)
            config_file = next(
                (f for f in files if f.path == "config.json"),
                files[0],  # fallback to first file
            )

            # Download the file - this exercises preflight validation
            content = sdk_with_s3_default.files.download_content(
                remote_path=config_file.path,
                fileset=fileset.name,
                workspace=fileset.workspace,
            )

            assert len(content) > 0, "Downloaded content should not be empty"

            # Verify it's valid JSON (config.json should be)
            if config_file.path == "config.json":
                import json

                parsed = json.loads(content)
                assert isinstance(parsed, dict), "config.json should be a JSON object"

        finally:
            files_client.delete_fileset(name=name, workspace=DEFAULT_WORKSPACE)
