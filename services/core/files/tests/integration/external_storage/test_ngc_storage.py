# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Integration tests for NGC storage backend.

These tests connect to the actual NVIDIA NGC and require network access.
They are skipped by default (see conftest.py in this directory).

To run these tests:

    RUN_EXTERNAL_STORAGE_TESTS=1 NGC_API_KEY=<your-key> pytest services/core/files/tests/integration/external_storage/

Prerequisites:
- NGC API key (via NGC_API_KEY environment variable)
- Access to NVIDIA NGC resources
"""

import asyncio
import os
import uuid
from typing import Iterator
from urllib.parse import parse_qs, urlparse

import pytest
from nemo_helix import NeMoHelix
from nemo_helix_plugin.client.adapter import client_from_platform
from nemo_helix_plugin.files.client import FilesClient
from nemo_helix_plugin.secrets.client import SecretsClient
from nemo_helix_plugin.secrets.types import PlatformSecretCreateRequest
from nhx.core.files.app.backends.base import StorageImpl
from nhx.core.files.app.streaming import download_url_streaming
from nhx.core.files.testing.utils import create_fileset
from pydantic import SecretStr

# Skip all tests in this module if NGC_API_KEY is not set
pytestmark = pytest.mark.skipif(
    not os.environ.get("NGC_API_KEY"),
    reason="NGC_API_KEY environment variable not set",
)

DEFAULT_WORKSPACE = "default"

# Pinned version for public NGC resource used in tests (nvidia/nemo-microservices/nemo-microservices-quickstart)
NGC_TEST_VERSION = "25.12"


@pytest.fixture
def ngc_api_key_secret(sdk: NeMoHelix) -> Iterator[str]:
    """Create a temporary secret for NGC API key and clean up after use."""
    api_key = os.environ.get("NGC_API_KEY")
    if not api_key:
        pytest.fail("NGC_API_KEY environment variable must be set")
    secret_name = f"ngc-api-key-{uuid.uuid4().hex[:8]}"
    secrets = client_from_platform(sdk, SecretsClient)
    secrets.create_secret(
        body=PlatformSecretCreateRequest(name=secret_name, value=SecretStr(api_key)),
        workspace=DEFAULT_WORKSPACE,
    )
    try:
        yield secret_name
    finally:
        secrets.delete_secret(name=secret_name, workspace=DEFAULT_WORKSPACE)


class TestNGCVersionResolution:
    """Test that mutable versions are resolved to immutable version IDs."""

    def test_fileset_resolves_latest_to_version_id(self, sdk: NeMoHelix, ngc_api_key_secret: str):
        """Test that creating a fileset without version resolves to the latest version ID.

        This verifies the fix for cache staleness: when a user creates a fileset
        without specifying a version (requesting 'latest'), the system should resolve
        it to the current version ID and store both values for auditing.
        """
        name = f"ngc-test-{uuid.uuid4().hex[:8]}"

        with create_fileset(
            sdk,
            name,
            storage={
                "type": "ngc",
                "org": "nvidia",
                "team": "nemo-microservices",
                "target": "nemo-microservices-quickstart",
                # No version specified - should resolve to latest
                "api_key_secret": ngc_api_key_secret,
            },
        ) as fileset:
            # Get the persisted fileset to check resolved values
            files = client_from_platform(sdk, FilesClient)
            persisted = files.get_fileset(
                name=fileset.name,
                workspace=fileset.workspace,
            ).data()

            storage = persisted.storage
            assert storage.type == "ngc"

            # version should be a specific version ID (not None)
            assert storage.version is not None, "version should be resolved to a specific ID"
            assert len(storage.version) > 0, "version should not be empty"

            # original_version should be None (user requested "latest")
            assert storage.original_version is None, f"original_version should be None, got: {storage.original_version}"

    def test_fileset_with_explicit_version_preserves_both(self, sdk: NeMoHelix, ngc_api_key_secret: str):
        """Test that creating a fileset with an explicit version preserves it correctly."""
        name = f"ngc-test-{uuid.uuid4().hex[:8]}"

        with create_fileset(
            sdk,
            name,
            storage={
                "type": "ngc",
                "org": "nvidia",
                "team": "nemo-microservices",
                "target": "nemo-microservices-quickstart",
                "version": NGC_TEST_VERSION,  # Explicit version
                "api_key_secret": ngc_api_key_secret,
            },
        ) as fileset:
            files = client_from_platform(sdk, FilesClient)
            persisted = files.get_fileset(
                name=fileset.name,
                workspace=fileset.workspace,
            ).data()

            storage = persisted.storage

            # Both should be the same version
            assert storage.version == NGC_TEST_VERSION
            assert storage.original_version == NGC_TEST_VERSION


class TestNGCStorageBackend:
    """Test NGC storage backend with real NGC resources."""

    def test_list_files_from_ngc_resource(self, sdk: NeMoHelix, ngc_api_key_secret: str):
        """Test listing files from an NGC resource."""
        name = f"ngc-test-{uuid.uuid4().hex[:8]}"

        with create_fileset(
            sdk,
            name,
            storage={
                "type": "ngc",
                "org": "nvidia",
                "team": "nemo-microservices",
                "target": "nemo-microservices-quickstart",
                "version": NGC_TEST_VERSION,
                "api_key_secret": ngc_api_key_secret,
            },
        ) as fileset:
            # List files from the NGC resource
            files = sdk.files.list(
                fileset=fileset.name,
                workspace=fileset.workspace,
            )

            # Should have files in the resource
            assert len(files.data) > 0

            # Verify file info has expected fields
            for file_info in files.data:
                assert file_info.path is not None
                assert file_info.size is not None
                assert file_info.size >= 0

    def test_download_file_from_ngc_resource(self, sdk: NeMoHelix, ngc_api_key_secret: str):
        """Test downloading a file from an NGC resource."""
        name = f"ngc-test-{uuid.uuid4().hex[:8]}"

        with create_fileset(
            sdk,
            name,
            storage={
                "type": "ngc",
                "org": "nvidia",
                "team": "nemo-microservices",
                "target": "nemo-microservices-quickstart",
                "version": NGC_TEST_VERSION,
                "api_key_secret": ngc_api_key_secret,
            },
        ) as fileset:
            # First list files to get a file path
            files = sdk.files.list(
                fileset=fileset.name,
                workspace=fileset.workspace,
            )
            assert len(files.data) > 0

            # Pick a small file to download (first one)
            test_file = files.data[0]

            # Download the file
            content = sdk.files.download_content(
                remote_path=test_file.path,
                fileset=fileset.name,
                workspace=fileset.workspace,
            )

            assert len(content) == test_file.size

    def test_download_with_range_request(self, sdk: NeMoHelix, ngc_api_key_secret: str):
        """Test partial download using HTTP Range header."""
        name = f"ngc-test-{uuid.uuid4().hex[:8]}"

        with create_fileset(
            sdk,
            name,
            storage={
                "type": "ngc",
                "org": "nvidia",
                "team": "nemo-microservices",
                "target": "nemo-microservices-quickstart",
                "version": NGC_TEST_VERSION,
                "api_key_secret": ngc_api_key_secret,
            },
        ) as fileset:
            # First list files to get a file path
            files = sdk.files.list(
                fileset=fileset.name,
                workspace=fileset.workspace,
            )
            assert len(files.data) > 0

            # Pick a file to test range request
            test_file = files.data[0]

            # Request first 1KB
            range_end = min(1023, test_file.size - 1)
            range_response = sdk.files._download_file(
                test_file.path,
                workspace=fileset.workspace,
                name=fileset.name,
                extra_headers={"Range": f"bytes=0-{range_end}"},
            )

            assert range_response.status_code == 206  # Partial Content
            range_content = range_response.read()
            expected_size = range_end + 1
            assert len(range_content) == expected_size


class TestNGCCaching:
    """Test that NGC downloads are properly cached."""

    def test_cache_path_uses_resolved_version_not_latest(
        self,
        sdk: NeMoHelix,
        ngc_api_key_secret: str,
        cache_storage_impl: StorageImpl,
    ):
        """Test that cache paths use resolved version ID, not 'latest'.

        This verifies the fix for cache staleness: cache paths should be based on
        the immutable version ID so that different versions don't share cache entries.
        """
        name = f"ngc-test-{uuid.uuid4().hex[:8]}"

        with create_fileset(
            sdk,
            name,
            storage={
                "type": "ngc",
                "org": "nvidia",
                "team": "nemo-microservices",
                "target": "nemo-microservices-quickstart",
                # No version specified - resolves to latest
                "api_key_secret": ngc_api_key_secret,
            },
        ) as fileset:
            # Get the resolved version ID
            files = client_from_platform(sdk, FilesClient)
            persisted = files.get_fileset(
                name=fileset.name,
                workspace=fileset.workspace,
            ).data()
            version_id = persisted.storage.version
            assert version_id is not None, "version should be resolved"

            # List files to get a file path
            files = sdk.files.list(
                fileset=fileset.name,
                workspace=fileset.workspace,
            )
            assert len(files.data) > 0
            test_file = files.data[0]

            # Download a file to populate the cache
            content = sdk.files.download_content(
                remote_path=test_file.path,
                fileset=fileset.name,
                workspace=fileset.workspace,
            )
            assert len(content) > 0

            # Cache path should use version ID
            # Format: cache/ngc/{org}/{team}/{resource}/{version}/{path}
            expected_cache_path = (
                f"cache/ngc/nvidia/nemo-microservices/nemo-microservices-quickstart/{version_id}/{test_file.path}"
            )

            cached_files = asyncio.run(cache_storage_impl.list_files(expected_cache_path))
            assert len(cached_files) == 1, f"File should be cached at version-specific path: {expected_cache_path}"

            # Verify cache path contains the version ID
            assert version_id in cached_files[0].path, (
                f"Cache path should contain version ID '{version_id}', got: {cached_files[0].path}"
            )

    def test_second_download_uses_cache(
        self,
        sdk: NeMoHelix,
        ngc_api_key_secret: str,
        cache_storage_impl: StorageImpl,
        mocker,
    ):
        """Test that the second download of the same file uses the cache."""

        name = f"ngc-test-{uuid.uuid4().hex[:8]}"

        # Spy on the source download function to verify it's only called on cache miss
        # Must patch where it's imported/used, not where it's defined
        download_spy = mocker.patch(
            "nhx.core.files.app.backends.ngc.download_url_streaming",
            wraps=download_url_streaming,
        )

        with create_fileset(
            sdk,
            name,
            storage={
                "type": "ngc",
                "org": "nvidia",
                "team": "nemo-microservices",
                "target": "nemo-microservices-quickstart",
                "version": NGC_TEST_VERSION,
                "api_key_secret": ngc_api_key_secret,
            },
        ) as fileset:
            # First list files to get a file path
            files = sdk.files.list(
                fileset=fileset.name,
                workspace=fileset.workspace,
            )
            assert len(files.data) > 0

            # Pick a small file to test
            test_file = files.data[0]

            # First download - should fetch from source (cache miss)
            content1 = sdk.files.download_content(
                remote_path=test_file.path,
                fileset=fileset.name,
                workspace=fileset.workspace,
            )

            # Source download should have been called twice:
            # 1. To stream content to the user
            # 2. In background task to cache the file
            assert download_spy.call_count == 2, "First download should fetch from source (serve + cache)"

            # Second download - should be served from cache (no source fetch)
            content2 = sdk.files.download_content(
                remote_path=test_file.path,
                fileset=fileset.name,
                workspace=fileset.workspace,
            )

            # Content should be identical
            assert content1 == content2

            # Source download should NOT have been called again
            assert download_spy.call_count == 2, "Second download should use cache, not fetch from source"

            # Verify the file was actually cached in the storage backend
            # Cache key format: cache/ngc/{org}/{team}/{resource}/{version}/{path}
            cache_path = (
                f"cache/ngc/nvidia/nemo-microservices/nemo-microservices-quickstart/{NGC_TEST_VERSION}/{test_file.path}"
            )
            cached_files = asyncio.run(cache_storage_impl.list_files(cache_path))
            assert len(cached_files) == 1, f"File should exist in cache at {cache_path}"

            # Verify cached content matches
            cached_file = cached_files[0]
            assert cached_file.path == cache_path
            assert cached_file.size == len(content1)

    def test_different_files_cached_separately(
        self,
        sdk: NeMoHelix,
        ngc_api_key_secret: str,
        cache_storage_impl: StorageImpl,
    ):
        """Test that different files from the same NGC resource are cached separately."""
        name = f"ngc-test-{uuid.uuid4().hex[:8]}"

        with create_fileset(
            sdk,
            name,
            storage={
                "type": "ngc",
                "org": "nvidia",
                "team": "nemo-microservices",
                "target": "nemo-microservices-quickstart",
                "version": NGC_TEST_VERSION,
                "api_key_secret": ngc_api_key_secret,
            },
        ) as fileset:
            # List files to get at least 2 file paths
            files = sdk.files.list(
                fileset=fileset.name,
                workspace=fileset.workspace,
            )
            assert len(files.data) >= 2, "Need at least 2 files for this test"

            file1 = files.data[0]
            file2 = files.data[1]

            # Download first file
            content1 = sdk.files.download_content(
                remote_path=file1.path,
                fileset=fileset.name,
                workspace=fileset.workspace,
            )

            # Download second file
            content2 = sdk.files.download_content(
                remote_path=file2.path,
                fileset=fileset.name,
                workspace=fileset.workspace,
            )

            # Files should be different
            assert file1.path != file2.path

            # Download first file again - should be from cache
            content3 = sdk.files.download_content(
                remote_path=file1.path,
                fileset=fileset.name,
                workspace=fileset.workspace,
            )
            assert content3 == content1

        # Verify both files exist in cache with correct paths
        cache_base = f"cache/ngc/nvidia/nemo-microservices/nemo-microservices-quickstart/{NGC_TEST_VERSION}/"

        # Check file1 is cached
        file1_cache_path = f"{cache_base}{file1.path}"
        file1_cached = asyncio.run(cache_storage_impl.list_files(file1_cache_path))
        assert len(file1_cached) == 1, f"{file1.path} should be cached at {file1_cache_path}"
        assert file1_cached[0].size == len(content1)

        # Check file2 is cached
        file2_cache_path = f"{cache_base}{file2.path}"
        file2_cached = asyncio.run(cache_storage_impl.list_files(file2_cache_path))
        assert len(file2_cached) == 1, f"{file2.path} should be cached at {file2_cache_path}"
        assert file2_cached[0].size == len(content2)

    def test_byte_range_requests_bypass_cache(
        self,
        sdk: NeMoHelix,
        ngc_api_key_secret: str,
        cache_storage_impl: StorageImpl,
    ):
        """Test that byte range requests bypass the cache but full downloads use cache."""
        name = f"ngc-test-{uuid.uuid4().hex[:8]}"

        with create_fileset(
            sdk,
            name,
            storage={
                "type": "ngc",
                "org": "nvidia",
                "team": "nemo-microservices",
                "target": "nemo-microservices-quickstart",
                "version": NGC_TEST_VERSION,
                "api_key_secret": ngc_api_key_secret,
            },
        ) as fileset:
            # Get a file to test with
            files = sdk.files.list(
                fileset=fileset.name,
                workspace=fileset.workspace,
            )
            assert len(files.data) > 0
            test_file = files.data[0]

            # First, do a full download to populate cache
            full_content = sdk.files.download_content(
                remote_path=test_file.path,
                fileset=fileset.name,
                workspace=fileset.workspace,
            )

            # Verify file is in cache after first download
            cache_path = (
                f"cache/ngc/nvidia/nemo-microservices/nemo-microservices-quickstart/{NGC_TEST_VERSION}/{test_file.path}"
            )
            cached_files = asyncio.run(cache_storage_impl.list_files(cache_path))
            assert len(cached_files) == 1, "File should be cached after full download"
            assert cached_files[0].size == len(full_content)

            # Now do a range request - should still work even though we have cache
            # Note: Range requests require the private _download_file method for extra_headers
            range_end = min(49, test_file.size - 1)
            range_response = sdk.files._download_file(
                test_file.path,
                workspace=fileset.workspace,
                name=fileset.name,
                extra_headers={"Range": f"bytes=0-{range_end}"},
            )
            assert range_response.status_code == 206  # Partial Content
            range_content = range_response.read()
            expected_size = range_end + 1
            assert len(range_content) == expected_size
            assert range_content == full_content[:expected_size]

            # Another full download should use cache
            full_content2 = sdk.files.download_content(
                remote_path=test_file.path,
                fileset=fileset.name,
                workspace=fileset.workspace,
            )
            assert full_content2 == full_content

            # Cache should still have exactly one file (no duplicate from byte range)
            cached_files_after = asyncio.run(cache_storage_impl.list_files(cache_path))
            assert len(cached_files_after) == 1, "Cache should not duplicate for byte range requests"


def _parse_ngc_catalog_url(url: str) -> tuple[str, str, str, str, str]:
    """Parse an NGC catalog URL into (org, team, target_name, version, target_type).

    Supports:
      .../orgs/{org}/teams/{team}/models/{name}?version=...
      .../orgs/{org}/teams/{team}/resources/{name}?version=...
      .../orgs/{org}/resources/{name}?version=...   (team is "no-team")

    Returns (org, team, target_name, version, target_type). target_type is "resource" or "model".
    """
    parsed = urlparse(url)
    if not parsed.path.startswith("/orgs/"):
        raise ValueError(f"Not an NGC catalog URL (expected path /orgs/...): {url!r}")
    segments = [s for s in parsed.path.split("/") if s]
    # ["orgs", org, "teams", team, "models"|"resources", name] or ["orgs", org, "resources", name]
    if len(segments) < 4 or segments[0] != "orgs":
        raise ValueError(f"Could not parse NGC catalog URL path: {url!r}")
    org = segments[1]
    version = ""
    if parsed.query:
        qs = parse_qs(parsed.query, strict_parsing=False)
        vers = qs.get("version", [])
        if vers:
            version = vers[0]

    if segments[2] == "teams":
        # .../orgs/{org}/teams/{team}/models/{name} or .../resources/{name}
        if len(segments) < 6:
            raise ValueError(f"Could not parse NGC catalog URL path: {url!r}")
        team = segments[3]
        type_seg = segments[4]
        target_name = segments[5] if len(segments) == 6 else "/".join(segments[5:])
        target_type = "model" if type_seg == "models" else "resource"
    elif segments[2] == "resources":
        # .../orgs/{org}/resources/{name}
        team = "no-team"
        target_name = segments[3] if len(segments) == 4 else "/".join(segments[3:])
        target_type = "resource"
    else:
        raise ValueError(f"Could not parse NGC catalog URL path: {url!r}")

    return org, team, target_name, version, target_type


# Public NGC catalog URLs for integration tests. Type (resource vs model) comes from the URL path.
NGC_PUBLIC_TARGETS_FOR_INTEGRATION = [
    "https://catalog.ngc.nvidia.com/orgs/nvidia/teams/nemo-microservices/resources/nemo-microservices-quickstart?version=25.12",
    "https://catalog.ngc.nvidia.com/orgs/nvidia/teams/earth-2/models/cbottle?version=1.2",
]


class TestNGCPublicTargets:
    """Integration tests using public NGC catalog URLs.

    Validates that the backend (ResourceAPI for resources, GuestModelAPI for models)
    can list files for public nvidia org assets when RUN_EXTERNAL_STORAGE_TESTS=1.
    """

    @pytest.mark.parametrize("catalog_url", NGC_PUBLIC_TARGETS_FOR_INTEGRATION)
    def test_list_files_public_ngc_target(self, sdk: NeMoHelix, ngc_api_key_secret: str, catalog_url: str):
        """List files from a public NGC resource or model using a catalog URL."""
        org, team, target_name, version, target_type = _parse_ngc_catalog_url(catalog_url)
        fileset_name = f"ngc-pub-{uuid.uuid4().hex[:8]}"

        storage = {
            "type": "ngc",
            "org": org,
            "team": team,
            "target": target_name,
            "target_type": target_type,
            "api_key_secret": ngc_api_key_secret,
        }
        if version:
            storage["version"] = version

        with create_fileset(sdk, fileset_name, storage=storage) as fileset:
            files = sdk.files.list(
                fileset=fileset.name,
                workspace=fileset.workspace,
            )
            assert len(files.data) > 0, "list_files should return at least one file"
            for file_info in files.data:
                assert file_info.path is not None
                assert file_info.size is not None
