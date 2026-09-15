# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Integration tests for Huggingface storage backend.

These tests connect to the actual Huggingface Hub and require network access.
They are skipped by default (see conftest.py in this directory).

To run these tests:

    RUN_EXTERNAL_STORAGE_TESTS=1 pytest services/core/files/tests/integration/external_storage/test_huggingface_storage.py
"""

import asyncio
import json
import os
import time
import uuid

import pytest
from filesets import FilesetFileSystem
from huggingface_hub import snapshot_download
from nemo_platform import NeMoPlatform
from nemo_platform_plugin.client.adapter import client_from_platform
from nemo_platform_plugin.client.errors import NemoHTTPError as ClientBadRequestError
from nemo_platform_plugin.files.client import FilesClient
from nemo_platform_plugin.files.types import CreateFilesetRequest, ListFilesQueryParams
from nmp.core.files.app.backends.base import StorageImpl
from nmp.core.files.app.streaming import download_url_streaming
from nmp.core.files.testing.utils import create_fileset


class TestHuggingfaceRevisionResolution:
    """Test that mutable revisions are resolved to immutable commit SHAs."""

    def test_fileset_resolves_main_to_commit_sha(self, sdk: NeMoPlatform):
        """Test that creating a fileset with revision='main' resolves to a commit SHA.

        This verifies the fix for cache staleness: when a user creates a fileset
        with a mutable reference like 'main', the system should resolve it to the
        current commit SHA and store both values for auditing.
        """
        name = f"hf-test-{uuid.uuid4().hex[:8]}"

        with create_fileset(
            sdk,
            name,
            storage={
                "type": "huggingface",
                "repo_id": "hf-internal-testing/tiny-random-bert",
                "repo_type": "model",
                "revision": "main",  # Mutable reference
            },
        ) as fileset:
            # Get the persisted fileset to check resolved values
            files = client_from_platform(sdk, FilesClient)
            persisted = files.get_fileset(
                name=fileset.name,
                workspace=fileset.workspace,
            ).data()

            storage = persisted.storage
            assert storage.type == "huggingface"

            # revision should be a 40-character commit SHA (not "main")
            assert storage.revision is not None
            assert len(storage.revision) == 40, f"Expected 40-char commit SHA, got: {storage.revision}"
            assert storage.revision != "main", "revision should be resolved SHA, not 'main'"
            # Should be hex characters only
            assert all(c in "0123456789abcdef" for c in storage.revision.lower()), (
                f"revision should be hex SHA, got: {storage.revision}"
            )

            # original_revision should preserve what user requested
            assert storage.original_revision == "main", (
                f"original_revision should be 'main', got: {storage.original_revision}"
            )

    def test_fileset_with_explicit_sha_preserves_both(self, sdk: NeMoPlatform):
        """Test that creating a fileset with an explicit SHA preserves it correctly."""
        name = f"hf-test-{uuid.uuid4().hex[:8]}"

        # First, get a valid commit SHA from the repo
        temp_name = f"hf-temp-{uuid.uuid4().hex[:8]}"
        with create_fileset(
            sdk,
            temp_name,
            storage={
                "type": "huggingface",
                "repo_id": "hf-internal-testing/tiny-random-bert",
                "repo_type": "model",
                "revision": "main",
            },
        ) as temp_fileset:
            files = client_from_platform(sdk, FilesClient)
            temp_persisted = files.get_fileset(
                name=temp_fileset.name,
                workspace=temp_fileset.workspace,
            ).data()
            commit_sha = temp_persisted.storage.revision

        # Now create a fileset with the explicit SHA
        with create_fileset(
            sdk,
            name,
            storage={
                "type": "huggingface",
                "repo_id": "hf-internal-testing/tiny-random-bert",
                "repo_type": "model",
                "revision": commit_sha,  # Explicit SHA
            },
        ) as fileset:
            persisted = files.get_fileset(
                name=fileset.name,
                workspace=fileset.workspace,
            ).data()

            storage = persisted.storage

            # Both should be the same SHA
            assert storage.revision == commit_sha
            assert storage.original_revision == commit_sha


class TestHuggingfaceStorageBackend:
    """Test Huggingface storage backend with real Huggingface Hub."""

    def test_list_files_from_public_dataset(self, sdk: NeMoPlatform):
        """Test listing files from a public Huggingface dataset."""
        name = f"hf-test-{uuid.uuid4().hex[:8]}"

        # Create fileset with Huggingface storage backend pointing to a small public dataset
        with create_fileset(
            sdk,
            name,
            storage={
                "type": "huggingface",
                "repo_id": "hf-internal-testing/tiny-random-bert",
                "repo_type": "model",
            },
        ) as fileset:
            # List files from the Huggingface repo
            files_response = (
                client_from_platform(sdk, FilesClient)
                .list_files(name=fileset.name, workspace=fileset.workspace)
                .data()
                .data
            )

            # Should have files in the repo
            assert len(files_response) > 0

            # Check for expected files (config.json is common in model repos)
            file_paths = {f.path for f in files_response}
            assert "config.json" in file_paths

    def test_gated_repo_fails_on_fileset_creation(self, sdk: NeMoPlatform):
        """Test that creating a fileset with a gated repo fails during validation.

        Gated repos like meta-llama/Llama-4-Scout-17B-16E-Instruct require access approval.
        The fileset creation should fail with a clear error message rather than
        succeeding and failing later on download.

        This test will fail if you use a HF_TOKEN that *does* have access to this model,
        so don't request access to this model!
        """
        name = f"hf-test-{uuid.uuid4().hex[:8]}"

        files = client_from_platform(sdk, FilesClient)
        with pytest.raises(ClientBadRequestError) as exc_info:
            files.create_fileset(
                workspace="default",
                body=CreateFilesetRequest(
                    name=name,
                    storage={
                        "type": "huggingface",
                        "repo_id": "meta-llama/Llama-4-Scout-17B-16E-Instruct",
                        "repo_type": "model",
                    },
                ),
            )

        # Should get a 400 error with access denied message
        assert exc_info.value.status_code == 400
        assert "Access denied" in str(exc_info.value) or "gated" in str(exc_info.value).lower()

    def test_download_file_from_public_dataset(self, sdk: NeMoPlatform):
        """Test downloading a file from a public Huggingface dataset."""
        name = f"hf-test-{uuid.uuid4().hex[:8]}"

        with create_fileset(
            sdk,
            name,
            storage={
                "type": "huggingface",
                "repo_id": "hf-internal-testing/tiny-random-bert",
                "repo_type": "model",
            },
        ) as fileset:
            # Download config.json
            content = (
                client_from_platform(sdk, FilesClient)
                .download_file(name=fileset.name, workspace=fileset.workspace, path="config.json")
                .read()
            )

            # Should be valid JSON
            config = json.loads(content)
            assert isinstance(config, dict)

    def test_download_with_range_request(self, sdk: NeMoPlatform):
        """Test partial download using HTTP Range header."""
        name = f"hf-test-{uuid.uuid4().hex[:8]}"

        with create_fileset(
            sdk,
            name,
            storage={
                "type": "huggingface",
                "repo_id": "hf-internal-testing/tiny-random-bert",
                "repo_type": "model",
            },
        ) as fileset:
            # First get full file to know its size
            files = client_from_platform(sdk, FilesClient)
            full_content = files.download_file(
                name=fileset.name, workspace=fileset.workspace, path="config.json"
            ).read()

            # Now request just the first 50 bytes using range header
            range_response = files.with_headers({"Range": "bytes=0-49"}).download_file(
                name=fileset.name, workspace=fileset.workspace, path="config.json"
            )
            range_content = range_response.read()
            assert range_response.http_response.status_code == 206  # Partial Content
            assert len(range_content) == 50
            assert range_content == full_content[:50]

    def test_file_exists_with_file_path(self, sdk: NeMoPlatform):
        """Test _exists with a file path returns True for existing files.

        This tests the fix for HuggingFace's list_repo_tree which expects directory
        paths. When _exists is called with a file path, list_files should fall back
        to get_file and return the file info, allowing _exists to return True.

        Regression test for: EntryNotFoundError when calling _exists with file path.
        """
        name = f"hf-test-{uuid.uuid4().hex[:8]}"

        with create_fileset(
            sdk,
            name,
            storage={
                "type": "huggingface",
                "repo_id": "hf-internal-testing/tiny-random-bert",
                "repo_type": "model",
            },
        ) as fileset:
            fs = FilesetFileSystem(client=client_from_platform(sdk, FilesClient))
            file_path = f"{fileset.workspace}/{fileset.name}#config.json"

            # This would fail with EntryNotFoundError before the fix
            exists = fs.exists(file_path)

            assert exists is True

    def test_file_exists_with_nonexistent_path_returns_false(
        self,
        sdk: NeMoPlatform,
    ):
        """Test _exists with a non-existent path returns False."""
        name = f"hf-test-{uuid.uuid4().hex[:8]}"

        with create_fileset(
            sdk,
            name,
            storage={
                "type": "huggingface",
                "repo_id": "hf-internal-testing/tiny-random-bert",
                "repo_type": "model",
            },
        ) as fileset:
            fs = FilesetFileSystem(client=client_from_platform(sdk, FilesClient))
            file_path = f"{fileset.workspace}/{fileset.name}#nonexistent/file/path.txt"

            # Should return False, not raise an error
            exists = fs.exists(file_path)

            assert exists is False

    def test_get_downloads_single_file(
        self,
        sdk: NeMoPlatform,
        tmp_path,
    ):
        """Test _get downloads a single file correctly.

        When source is a single file path, the file should be downloaded
        to dest/filename.
        """

        name = f"hf-test-{uuid.uuid4().hex[:8]}"

        with create_fileset(
            sdk,
            name,
            storage={
                "type": "huggingface",
                "repo_id": "hf-internal-testing/tiny-random-bert",
                "repo_type": "model",
            },
        ) as fileset:
            fs = FilesetFileSystem(client=client_from_platform(sdk, FilesClient))
            file_path = f"{fileset.workspace}/{fileset.name}#config.json"

            # Download single file
            fs.get(file_path, str(tmp_path))

            # File should be at tmp_path/config.json
            downloaded_file = tmp_path / "config.json"
            assert downloaded_file.exists()

            # Should be valid JSON
            content = json.loads(downloaded_file.read_text())
            assert isinstance(content, dict)

    def test_get_downloads_directory_with_trailing_slash(
        self,
        sdk: NeMoPlatform,
        tmp_path,
    ):
        """Test _get with trailing slash copies contents directly into dest.

        When source has trailing /, the directory contents should be copied
        directly into the destination without preserving the source dir name.
        """

        name = f"hf-test-{uuid.uuid4().hex[:8]}"

        with create_fileset(
            sdk,
            name,
            storage={
                "type": "huggingface",
                "repo_id": "hf-internal-testing/tiny-random-bert",
                "repo_type": "model",
            },
        ) as fileset:
            fs = FilesetFileSystem(client=client_from_platform(sdk, FilesClient))
            # Trailing slash on source - copy contents directly
            dir_path = f"{fileset.workspace}/{fileset.name}#/"

            fs.get(dir_path, str(tmp_path))

            # config.json should be directly in tmp_path (not tmp_path/<fileset_name>/)
            assert (tmp_path / "config.json").exists()

    def test_get_downloads_directory_without_trailing_slash(
        self,
        sdk: NeMoPlatform,
        tmp_path,
    ):
        """Test _get for fileset root copies contents directly.

        For fileset root (no file path), contents are always copied directly
        regardless of trailing slash. This matches common tools like AWS S3,
        rclone, etc. Users who want a subfolder can include it in local_path.
        """

        name = f"hf-test-{uuid.uuid4().hex[:8]}"

        with create_fileset(
            sdk,
            name,
            storage={
                "type": "huggingface",
                "repo_id": "hf-internal-testing/tiny-random-bert",
                "repo_type": "model",
            },
        ) as fileset:
            fs = FilesetFileSystem(client=client_from_platform(sdk, FilesClient))
            # No trailing slash on source - for fileset root, copies contents directly
            dir_path = f"{fileset.workspace}/{fileset.name}#"

            fs.get(dir_path, str(tmp_path))

            # Files should be directly in tmp_path/ (no fileset subfolder)
            assert (tmp_path / "config.json").exists()
            assert not (tmp_path / fileset.name).exists()


class TestHuggingfaceCaching:
    """Test that HuggingFace downloads are properly cached."""

    def test_cache_path_uses_resolved_sha_not_mutable_ref(self, sdk: NeMoPlatform, cache_storage_impl: StorageImpl):
        """Test that cache paths use resolved commit SHA, not mutable refs like 'main'.

        This verifies the fix for cache staleness: cache paths should be based on
        the immutable commit SHA so that:
        1. Different commits don't share cache entries (prevents stale data)
        2. Same content across commits CAN share cache (content-addressed deduplication)
        """
        name = f"hf-test-{uuid.uuid4().hex[:8]}"

        with create_fileset(
            sdk,
            name,
            storage={
                "type": "huggingface",
                "repo_id": "hf-internal-testing/tiny-random-bert",
                "repo_type": "model",
                "revision": "main",  # Mutable reference
            },
        ) as fileset:
            # Get the resolved commit SHA
            files = client_from_platform(sdk, FilesClient)
            persisted = files.get_fileset(
                name=fileset.name,
                workspace=fileset.workspace,
            ).data()
            commit_sha = persisted.storage.revision
            assert commit_sha != "main", "revision should be resolved to SHA"

            # Download a file to populate the cache
            content = (
                client_from_platform(sdk, FilesClient)
                .download_file(name=fileset.name, workspace=fileset.workspace, path="config.json")
                .read()
            )
            assert len(content) > 0

            # Cache path should use commit SHA, NOT "main"
            # Format: cache/hf/{repo_id}/{commit_sha}/{path}

            # Check that "main" is NOT in any cache path
            all_cached = asyncio.run(cache_storage_impl.list_files("cache/hf/"))

            for cached_file in all_cached:
                assert "/main/" not in cached_file.path, (
                    f"Cache path should not contain '/main/', got: {cached_file.path}"
                )

            # Verify file IS cached under repo_id with commit SHA as identifier
            # Format: cache/hf/{repo_id}/{commit_sha}/{path}
            repo_cached = asyncio.run(cache_storage_impl.list_files("cache/hf/hf-internal-testing/tiny-random-bert/"))

            config_cached = [f for f in repo_cached if "config.json" in f.path]
            assert len(config_cached) >= 1, (
                f"config.json should be cached under repo path. Found: {[f.path for f in repo_cached]}"
            )

    def test_second_download_uses_cache(self, sdk: NeMoPlatform, cache_storage_impl: StorageImpl, mocker):
        """Test that the second download of the same file uses the cache."""

        name = f"hf-test-{uuid.uuid4().hex[:8]}"

        # Spy on the source download function to verify it's only called on cache miss
        # Must patch where it's imported/used, not where it's defined
        download_spy = mocker.patch(
            "nmp.core.files.app.backends.huggingface.download_url_streaming",
            wraps=download_url_streaming,
        )

        with create_fileset(
            sdk,
            name,
            storage={
                "type": "huggingface",
                "repo_id": "hf-internal-testing/tiny-random-bert",
                "repo_type": "model",
                "revision": "main",  # Explicit revision for cache key
            },
        ) as fileset:
            # First download - should fetch from source (cache miss)
            content1 = (
                client_from_platform(sdk, FilesClient)
                .download_file(name=fileset.name, workspace=fileset.workspace, path="config.json")
                .read()
            )

            # Source download should have been called twice:
            # 1. To stream content to the user
            # 2. In background task to cache the file
            assert download_spy.call_count == 2, "First download should fetch from source (serve + cache)"

            # Second download - should be served from cache (no source fetch)
            content2 = (
                client_from_platform(sdk, FilesClient)
                .download_file(name=fileset.name, workspace=fileset.workspace, path="config.json")
                .read()
            )

            # Content should be identical
            assert content1 == content2

            # Source download should NOT have been called again
            assert download_spy.call_count == 2, "Second download should use cache, not fetch from source"

            # Verify the file was actually cached in the storage backend
            # Format: cache/hf/{repo_id}/{commit_sha}/{path}
            repo_cached = asyncio.run(cache_storage_impl.list_files("cache/hf/hf-internal-testing/tiny-random-bert/"))
            config_cached = [f for f in repo_cached if "config.json" in f.path]

            assert len(config_cached) >= 1, f"config.json should be cached. Found: {[f.path for f in repo_cached]}"

            # Verify cached content matches
            cached_file = config_cached[0]
            assert cached_file.size == len(content1)

    def test_different_files_cached_separately(self, sdk: NeMoPlatform, cache_storage_impl: StorageImpl):
        """Test that different files from the same repo are cached separately."""
        name = f"hf-test-{uuid.uuid4().hex[:8]}"

        with create_fileset(
            sdk,
            name,
            storage={
                "type": "huggingface",
                "repo_id": "hf-internal-testing/tiny-random-bert",
                "repo_type": "model",
                "revision": "main",
            },
        ) as fileset:
            # Download config.json
            config_content = (
                client_from_platform(sdk, FilesClient)
                .download_file(name=fileset.name, workspace=fileset.workspace, path="config.json")
                .read()
            )

            # Download tokenizer_config.json (different file)
            tokenizer_content = (
                client_from_platform(sdk, FilesClient)
                .download_file(name=fileset.name, workspace=fileset.workspace, path="tokenizer_config.json")
                .read()
            )

            # Files should be different
            assert config_content != tokenizer_content

            # Download config.json again - should be from cache
            config_content2 = (
                client_from_platform(sdk, FilesClient)
                .download_file(name=fileset.name, workspace=fileset.workspace, path="config.json")
                .read()
            )
            assert config_content2 == config_content

        # Verify both files exist in cache
        # Format: cache/hf/{repo_id}/{commit_sha}/{path}
        repo_cached = asyncio.run(cache_storage_impl.list_files("cache/hf/hf-internal-testing/tiny-random-bert/"))

        # Check config.json is cached (use endswith to avoid matching tokenizer_config.json)
        config_cached = [f for f in repo_cached if f.path.endswith("/config.json")]
        assert len(config_cached) >= 1, f"config.json should be cached. Found: {[f.path for f in repo_cached]}"
        assert config_cached[0].size == len(config_content)

        # Check tokenizer_config.json is cached
        tokenizer_cached = [f for f in repo_cached if f.path.endswith("/tokenizer_config.json")]
        assert len(tokenizer_cached) >= 1, (
            f"tokenizer_config.json should be cached. Found: {[f.path for f in repo_cached]}"
        )
        assert tokenizer_cached[0].size == len(tokenizer_content)

    def test_byte_range_requests_bypass_cache(self, sdk: NeMoPlatform, cache_storage_impl: StorageImpl):
        """Test that byte range requests bypass the cache but full downloads use cache."""
        name = f"hf-test-{uuid.uuid4().hex[:8]}"

        with create_fileset(
            sdk,
            name,
            storage={
                "type": "huggingface",
                "repo_id": "hf-internal-testing/tiny-random-bert",
                "repo_type": "model",
                "revision": "main",
            },
        ) as fileset:
            # First, do a full download to populate cache
            files = client_from_platform(sdk, FilesClient)
            full_content = files.download_file(
                name=fileset.name, workspace=fileset.workspace, path="config.json"
            ).read()

            # Verify file is in cache after first download
            # Format: cache/hf/{repo_id}/{commit_sha}/{path}
            repo_cached = asyncio.run(cache_storage_impl.list_files("cache/hf/hf-internal-testing/tiny-random-bert/"))
            config_cached = [f for f in repo_cached if "config.json" in f.path]
            assert len(config_cached) >= 1, "File should be cached after full download"
            assert config_cached[0].size == len(full_content)

            # Now do a range request - should still work even though we have cache
            range_response = files.with_headers({"Range": "bytes=0-49"}).download_file(
                name=fileset.name, workspace=fileset.workspace, path="config.json"
            )
            range_content = range_response.read()
            assert range_response.http_response.status_code == 206  # Partial Content
            assert len(range_content) == 50
            assert range_content == full_content[:50]

            # Another full download should use cache
            full_content2 = files.download_file(
                name=fileset.name, workspace=fileset.workspace, path="config.json"
            ).read()
            assert full_content2 == full_content

            # Cache should still have exactly one file (no duplicate from byte range)
            repo_cached_after = asyncio.run(
                cache_storage_impl.list_files("cache/hf/hf-internal-testing/tiny-random-bert/")
            )
            config_after = [f for f in repo_cached_after if "config.json" in f.path]
            assert len(config_after) == 1, "Cache should not duplicate for byte range requests"

    def test_cache_warming_on_create(self, sdk: NeMoPlatform, cache_storage_impl: StorageImpl):
        """Test that cache=True warms cache on fileset creation."""
        name = f"hf-test-{uuid.uuid4().hex[:8]}"

        with create_fileset(
            sdk,
            name,
            storage={
                "type": "huggingface",
                "repo_id": "hf-internal-testing/tiny-random-bert",
                "repo_type": "model",
                "revision": "main",
            },
            cache=True,
        ) as fileset:
            # Poll until files are cached or timeout
            max_attempts = 30
            for _ in range(max_attempts):
                files_response = (
                    client_from_platform(sdk, FilesClient)
                    .list_files(
                        name=fileset.name,
                        workspace=fileset.workspace,
                        query_params=ListFilesQueryParams(include_cache_status=True),
                    )
                    .data()
                    .data
                )

                # Check if all files are cached
                statuses = {f.cache_status for f in files_response}
                if statuses == {"cached"}:
                    break

                time.sleep(0.5)
            else:
                pytest.fail(f"Cache warming timed out after {max_attempts * 0.5}s. Final statuses: {statuses}")

            # Verify all files are cached
            for f in files_response:
                assert f.cache_status == "cached", f"File {f.path} should be cached, got {f.cache_status}"

    def test_cache_warming_disabled_by_default(self, sdk: NeMoPlatform, cache_storage_impl: StorageImpl):
        """Test that cache=False (default) does not warm cache."""
        name = f"hf-test-{uuid.uuid4().hex[:8]}"

        with create_fileset(
            sdk,
            name,
            storage={
                "type": "huggingface",
                "repo_id": "hf-internal-testing/tiny-random-bert",
                "repo_type": "model",
                "revision": "main",
            },
            # cache=False is the default
        ) as fileset:
            # Give a moment for any background task to start (shouldn't happen)
            time.sleep(0.5)

            # Check cache status - files should NOT be cached
            files_response = (
                client_from_platform(sdk, FilesClient)
                .list_files(
                    name=fileset.name,
                    workspace=fileset.workspace,
                    query_params=ListFilesQueryParams(include_cache_status=True),
                )
                .data()
                .data
            )

            # All files should be not_cached (no warming happened)
            for f in files_response:
                assert f.cache_status == "not_cached", f"File {f.path} should not be cached without cache=True"


class TestHuggingfaceHubClientCompatibility:
    """Test compatibility with the huggingface_hub client library.

    These tests verify that the HF-compat API endpoints work correctly
    with huggingface_hub's snapshot_download and other methods when used
    with external HuggingFace storage backends.
    """

    def test_snapshot_download_via_hf_compat_api(self, sdk: NeMoPlatform, tmp_path, hf_asgi_client):
        """Test downloading a fileset using huggingface_hub's snapshot_download.

        This validates that the HF-compat API endpoints (/v2/hf/...) work correctly
        with the huggingface_hub client library, allowing users to download filesets
        backed by external HuggingFace repos using familiar HuggingFace tooling.
        """
        name = f"hf-test-{uuid.uuid4().hex[:8]}"

        with create_fileset(
            sdk,
            name,
            storage={
                "type": "huggingface",
                "repo_id": "hf-internal-testing/tiny-random-bert",
                "repo_type": "model",
            },
        ) as fileset:
            # Get the base URL from the SDK's httpx client
            base_url = str(sdk._client.base_url).rstrip("/")

            # Use huggingface_hub's snapshot_download with our HF-compat endpoint
            local_dir = snapshot_download(
                repo_id=f"{fileset.workspace}/{fileset.name}",
                endpoint=f"{base_url}/apis/files/v2/hf",
                local_dir=str(tmp_path / "downloaded"),
                token="service:test",  # Service principal token required by HF-compat endpoints
            )

            # Verify files were downloaded
            assert os.path.exists(local_dir)
            assert os.path.isfile(os.path.join(local_dir, "config.json"))

            # Verify content is valid JSON
            with open(os.path.join(local_dir, "config.json")) as f:
                config = json.load(f)
            assert isinstance(config, dict)

    def test_snapshot_download_creates_correct_structure(self, sdk: NeMoPlatform, tmp_path, hf_asgi_client):
        """Test that snapshot_download preserves the repository file structure."""
        name = f"hf-test-{uuid.uuid4().hex[:8]}"

        with create_fileset(
            sdk,
            name,
            storage={
                "type": "huggingface",
                "repo_id": "hf-internal-testing/tiny-random-bert",
                "repo_type": "model",
            },
        ) as fileset:
            base_url = str(sdk._client.base_url).rstrip("/")

            # First, list files to know what to expect
            files_response = (
                client_from_platform(sdk, FilesClient).list_files(name=fileset.name, workspace=fileset.workspace).data()
            )
            expected_files = {f.path for f in files_response}

            # Download via snapshot_download
            local_dir = snapshot_download(
                repo_id=f"{fileset.workspace}/{fileset.name}",
                endpoint=f"{base_url}/apis/files/v2/hf",
                local_dir=str(tmp_path / "repo"),
                token="service:test",  # Service principal token required by HF-compat endpoints
            )

            # Verify all expected files were downloaded
            for expected_file in expected_files:
                local_path = os.path.join(local_dir, expected_file)
                assert os.path.exists(local_path), f"Expected file {expected_file} not found at {local_path}"

    def test_download_config_files_excluding_large_model_files(self, sdk: NeMoPlatform, tmp_path):
        """Test downloading only config files from a model repo, excluding large model files.

        This test demonstrates:
        1. Listing all files in the repo
        2. Filtering to get only small config files (excluding large model files)
        3. Downloading just those files using sdk.files.download with a list of paths
        4. Download creates necessary directories when local_path doesn't exist
        """
        name = f"hf-test-{uuid.uuid4().hex[:8]}"

        # Large model file suffixes to exclude
        large_file_suffixes = (
            ".safetensor",
            ".safetensors",
            ".bin",
            ".pkl",
            ".npy",
            ".onnx",
            ".pth",
        )

        with create_fileset(
            sdk,
            name,
            storage={
                "type": "huggingface",
                "repo_id": "openai/gpt-oss-20b",
                "repo_type": "model",
            },
        ) as fileset:
            # 1. List all files in the repo
            all_files = (
                client_from_platform(sdk, FilesClient)
                .list_files(name=fileset.name, workspace=fileset.workspace)
                .data()
                .data
            )

            # 2. Filter to get only config files (exclude large model files)
            config_only_paths = [f.path for f in all_files if not f.path.endswith(large_file_suffixes)]

            assert len(config_only_paths) > 0, "Should have some config files"

            # 3. Download each config file to a nested path that doesn't exist yet
            downloads = client_from_platform(sdk, FilesClient)
            download_dir = tmp_path / "nested" / "path" / "downloads"
            assert not download_dir.exists(), "Directory should not exist before download"

            for path in config_only_paths:
                dest = download_dir / path
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_bytes(
                    downloads.download_file(name=fileset.name, workspace=fileset.workspace, path=path).read()
                )

            # 4. Verify directory was created and files were downloaded
            assert download_dir.exists(), "Directory should be created by download"

            downloaded_files = list(download_dir.rglob("*"))
            downloaded_files = [f for f in downloaded_files if f.is_file()]

            assert len(downloaded_files) == len(config_only_paths), (
                f"Expected {len(config_only_paths)} files, got {len(downloaded_files)}"
            )

            # 5. Verify directory hierarchy is preserved (not flat downloads)
            nested_paths = [p for p in config_only_paths if "/" in p]
            assert len(nested_paths) > 0, "Test requires at least one nested file path to verify hierarchy preservation"
            for nested_path in nested_paths:
                expected_file = download_dir / nested_path
                assert expected_file.exists(), (
                    f"Nested file should be at {expected_file}, not flattened. Directory hierarchy must be preserved."
                )

            # Verify no large model files were downloaded
            for f in downloaded_files:
                assert not f.name.endswith(large_file_suffixes), f"Should not have downloaded large model file: {f}"
