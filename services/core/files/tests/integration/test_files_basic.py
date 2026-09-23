# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Integration tests for the files service API.

These tests verify:
- Fileset CRUD operations
- File upload/download (multipart and octet-stream)
- HTTP range requests
- DuckDB httpfs integration

Uses the create_test_client pattern for fast in-memory testing.
"""

import concurrent.futures
import time
from contextlib import ExitStack
from datetime import datetime, timedelta, timezone
from pathlib import Path

import duckdb
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from nemo_helix import NeMoHelix
from nemo_helix_plugin.client.adapter import client_from_platform
from nemo_helix_plugin.client.errors import BadRequestError, ConflictError, NemoHTTPError, NotFoundError
from nemo_helix_plugin.files.client import FilesClient
from nemo_helix_plugin.files.types import CreateFilesetRequest, FilesetOutput, UpdateFilesetRequest
from nhx.core.files.testing.utils import (
    DEFAULT_WORKSPACE_ID,
    HTTPXFileSystem,
    create_fileset,
)
from pydantic import ValidationError


class TestFilesBasic:
    def test_fileset_get(self, sdk: NeMoHelix):
        files = client_from_platform(sdk, FilesClient)
        with create_fileset(sdk) as fileset:
            fetched = files.get_fileset(name=fileset.name, workspace=fileset.workspace).data()
            assert fetched.id == fileset.id
            assert fetched.name == fileset.name

    def test_fileset_list(self, sdk: NeMoHelix):
        """Test listing filesets and filtering by workspace."""
        files = client_from_platform(sdk, FilesClient)
        with create_fileset(sdk) as fileset1:
            with create_fileset(sdk) as fileset2:
                filesets = list(files.list_filesets(workspace=DEFAULT_WORKSPACE_ID).items())
                assert any(fs.id == fileset1.id for fs in filesets)
                assert any(fs.id == fileset2.id for fs in filesets)

    def test_fileset_list_filter_by_name(self, sdk: NeMoHelix):
        """Test listing filesets with name filter."""
        files = client_from_platform(sdk, FilesClient)
        with create_fileset(sdk) as fileset1:
            with create_fileset(sdk) as fileset2:
                # Filter by exact name of fileset1
                filtered = list(
                    files.list_filesets(
                        workspace=DEFAULT_WORKSPACE_ID,
                        query_params={"filter": {"name": fileset1.name}},
                    ).items()
                )
                assert len(filtered) == 1
                assert filtered[0].id == fileset1.id
                assert filtered[0].name == fileset1.name

                # Filter by exact name of fileset2
                filtered2 = list(
                    files.list_filesets(
                        workspace=DEFAULT_WORKSPACE_ID,
                        query_params={"filter": {"name": fileset2.name}},
                    ).items()
                )
                assert len(filtered2) == 1
                assert filtered2[0].id == fileset2.id

                # Filter by non-existent name should return empty
                filtered_none = list(
                    files.list_filesets(
                        workspace=DEFAULT_WORKSPACE_ID,
                        query_params={"filter": {"name": "non-existent-fileset-name"}},
                    ).items()
                )
                assert len(filtered_none) == 0

    def test_fileset_list_filter_by_purpose(self, sdk: NeMoHelix):
        """Test listing filesets with purpose filter."""
        files = client_from_platform(sdk, FilesClient)
        with create_fileset(sdk, purpose="dataset") as dataset_fileset:
            with create_fileset(sdk, purpose="generic") as generic_fileset:
                # Filter by purpose=dataset
                dataset_filesets = list(
                    files.list_filesets(
                        workspace=DEFAULT_WORKSPACE_ID,
                        query_params={"filter": {"purpose": "dataset"}},
                    ).items()
                )
                assert any(fs.id == dataset_fileset.id for fs in dataset_filesets)
                assert not any(fs.id == generic_fileset.id for fs in dataset_filesets)

                # Filter by purpose=generic
                generic_filesets = list(
                    files.list_filesets(
                        workspace=DEFAULT_WORKSPACE_ID,
                        query_params={"filter": {"purpose": "generic"}},
                    ).items()
                )
                assert any(fs.id == generic_fileset.id for fs in generic_filesets)
                assert not any(fs.id == dataset_fileset.id for fs in generic_filesets)

    def test_fileset_list_filter_by_storage_type(self, sdk: NeMoHelix):
        """Test listing filesets with storage_type filter."""
        files = client_from_platform(sdk, FilesClient)
        # Create filesets with default local storage
        with create_fileset(sdk) as local_fileset1:
            with create_fileset(sdk) as local_fileset2:
                # Filter by storage_type=local
                local_filesets = list(
                    files.list_filesets(
                        workspace=DEFAULT_WORKSPACE_ID,
                        query_params={"filter": {"storage_type": "local"}},
                    ).items()
                )
                assert any(fs.id == local_fileset1.id for fs in local_filesets)
                assert any(fs.id == local_fileset2.id for fs in local_filesets)
                # Verify storage type is local
                for fs in local_filesets:
                    if fs.id in [local_fileset1.id, local_fileset2.id]:
                        assert fs.storage.type == "local"

    def test_fileset_list_pagination(self, sdk: NeMoHelix):
        """Test listing filesets with pagination."""
        files = client_from_platform(sdk, FilesClient)
        with ExitStack() as stack:
            # Create 5 filesets using ExitStack for automatic cleanup
            for _ in range(5):
                stack.enter_context(create_fileset(sdk, purpose="generic"))

            # Test first page with page_size=2
            resp1 = files.list_filesets(
                workspace=DEFAULT_WORKSPACE_ID,
                query_params={"page": 1, "page_size": 2},
            )
            page1 = resp1.page()
            assert len(page1.items) == 2
            assert page1.metadata["page"] == 1
            assert page1.metadata["page_size"] == 2

            # Test second page
            resp2 = files.list_filesets(
                workspace=DEFAULT_WORKSPACE_ID,
                query_params={"page": 2, "page_size": 2},
            )
            page2 = resp2.page()
            assert len(page2.items) == 2
            assert page2.metadata["page"] == 2

            # Verify pages have different data
            page1_ids = {fs.id for fs in page1.items}
            page2_ids = {fs.id for fs in page2.items}
            assert page1_ids.isdisjoint(page2_ids), "Pages should have different filesets"

    def test_file_upload_download(self, sdk: NeMoHelix, fileset: FilesetOutput):
        """Test uploading and downloading a file using application/octet-stream."""

        test_content = b"Hello, World! This is a test file.\nLine 2\nLine 3"
        test_path = "test.txt"

        files = client_from_platform(sdk, FilesClient)
        files.upload_file(
            content=test_content,
            path=test_path,
            name=fileset.name,
            workspace=fileset.workspace,
        )

        # Verify upload succeeded
        files_response = (
            client_from_platform(sdk, FilesClient)
            .list_files(name=fileset.name, workspace=fileset.workspace)
            .data()
            .data
        )
        assert len(files_response) == 1
        uploaded_file = files_response[0]
        assert uploaded_file.path == test_path
        assert uploaded_file.size == len(test_content)

        # Download file
        downloaded = (
            client_from_platform(sdk, FilesClient)
            .download_file(name=fileset.name, workspace=fileset.workspace, path="test.txt")
            .read()
        )
        assert downloaded == test_content

        files.delete_file(
            path=test_path,
            name=fileset.name,
            workspace=fileset.workspace,
        )

        files_response = (
            client_from_platform(sdk, FilesClient)
            .list_files(name=fileset.name, workspace=fileset.workspace)
            .data()
            .data
        )
        assert len(files_response) == 0, "File should be deleted"

    def test_file_upload_nested_paths_and_list(self, sdk: NeMoHelix, fileset: FilesetOutput):
        """Test uploading multiple files with nested paths concurrently and listing them."""

        # Upload multiple files with nested paths
        test_files = {
            "root.txt": b"Root level file",
            "folder1/file1.txt": b"File in folder1",
            "folder1/file2.txt": b"Another file in folder1",
            "folder1/subfolder/nested.txt": b"Nested file",
            "folder2/data.txt": b"File in folder2",
        }

        def upload_file(path_content_tuple):
            """Upload a single file."""
            path, content = path_content_tuple
            client_from_platform(sdk, FilesClient).upload_file(
                content=content,
                path=path,
                name=fileset.name,
                workspace=fileset.workspace,
            )
            return path

        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
            futures = {executor.submit(upload_file, item): item for item in test_files.items()}

            for future in concurrent.futures.as_completed(futures):
                path = future.result()
                assert path in test_files

        files_response = (
            client_from_platform(sdk, FilesClient)
            .list_files(name=fileset.name, workspace=fileset.workspace)
            .data()
            .data
        )

        listed_paths = {f.path for f in files_response}
        assert listed_paths == set(test_files.keys())

        # Verify each file has correct size
        for f in files_response:
            assert f.size == len(test_files[f.path])

        for path, expected_content in test_files.items():
            downloaded = (
                client_from_platform(sdk, FilesClient)
                .download_file(name=fileset.name, workspace=fileset.workspace, path=path)
                .read()
            )
            assert downloaded == expected_content

    def test_file_range_requests_with_duckdb(self, sdk: NeMoHelix, fileset: FilesetOutput, client: TestClient):
        """Test HTTP range requests by querying a parquet file with DuckDB.

        Uses HTTPXFileSystem to route DuckDB requests through the test client,
        enabling in-memory testing without a real HTTP server.
        """

        # Create a simple parquet file with test data
        test_data = pd.DataFrame(
            {
                "id": [1, 2, 3, 4, 5],
                "name": ["Alice", "Bob", "Charlie", "David", "Eve"],
                "age": [25, 30, 35, 40, 45],
                "city": ["New York", "London", "Paris", "Tokyo", "Sydney"],
            }
        )

        parquet_bytes = test_data.to_parquet(index=False)
        parquet_path = "test_data.parquet"

        # Upload parquet file
        client_from_platform(sdk, FilesClient).upload_file(
            content=parquet_bytes,
            path=parquet_path,
            name=fileset.name,
            workspace=fileset.workspace,
        )
        # Get the file info for the file_url
        files = (
            client_from_platform(sdk, FilesClient)
            .list_files(name=fileset.name, workspace=fileset.workspace)
            .data()
            .data
        )
        upload_response = next(f for f in files if f.path == parquet_path)

        # Use DuckDB with HTTPXFileSystem to query via the test client's transport
        conn = duckdb.connect(":memory:")
        httpx_fs = HTTPXFileSystem(client=client)
        conn.register_filesystem(httpx_fs)

        # Query the parquet file using the 'httpx' protocol with the file_url path
        file_path = upload_response.file_url
        result = conn.execute(f"SELECT * FROM read_parquet('httpx://{file_path}')").fetchdf()

        # Verify query results match original data
        pd.testing.assert_frame_equal(result, test_data)

        # Test a filtered query to verify range requests work
        filtered_result = conn.execute(
            f"SELECT name, age FROM read_parquet('httpx://{file_path}') WHERE age > 30"
        ).fetchdf()

        expected_filtered = test_data[test_data["age"] > 30][["name", "age"]].reset_index(drop=True)
        pd.testing.assert_frame_equal(filtered_result, expected_filtered)

        conn.close()

    def test_error_handling(self, sdk: NeMoHelix):
        """Test error handling for various 404 scenarios."""
        files = client_from_platform(sdk, FilesClient)

        # Test 1: Get non-existent fileset
        try:
            files.get_fileset(
                name="non-existent-fileset",
                workspace="non-existent-workspace",
            )
            assert False, "Should have raised NotFoundError"
        except NotFoundError:
            pass  # Expected

        # Test 2: Delete fileset and verify subsequent get returns 404
        with create_fileset(sdk) as fileset:
            fileset_name_str = fileset.name
            workspace_str = fileset.workspace

            # Upload a file to verify fileset exists
            files.upload_file(
                content=b"test content",
                path="test.txt",
                name=fileset_name_str,
                workspace=workspace_str,
            )

            # Verify fileset exists
            retrieved = files.get_fileset(
                name=fileset_name_str,
                workspace=workspace_str,
            ).data()
            assert retrieved.id == fileset.id

        # After context manager exits, fileset is deleted
        # Verify getting the fileset now raises 404
        try:
            files.get_fileset(
                name=fileset_name_str,
                workspace=workspace_str,
            )
            assert False, "Should have raised NotFoundError after fileset deletion"
        except NotFoundError:
            pass  # Expected

        # Test 3: Try to download non-existent file
        with create_fileset(sdk) as fileset:
            try:
                files.download_file(
                    path="non-existent-file.txt",
                    name=fileset.name,
                    workspace=fileset.workspace,
                ).read()
                assert False, "Should have raised NotFoundError for non-existent file"
            except NotFoundError:
                pass  # Expected

        # Test 4: Try to delete non-existent file
        with create_fileset(sdk) as fileset:
            try:
                files.delete_file(
                    path="non-existent-file.txt",
                    name=fileset.name,
                    workspace=fileset.workspace,
                )
                assert False, "Should have raised NotFoundError when deleting non-existent file"
            except NotFoundError:
                pass  # Expected

        # Test 5: List files in non-existent fileset
        try:
            files.list_files(
                name="non-existent-fileset",
                workspace="non-existent-workspace",
            ).data()
            assert False, "Should have raised NotFoundError"
        except NotFoundError:
            pass  # Expected

    def test_fileset_create_conflict(self, sdk: NeMoHelix):
        """Test that creating a fileset with a duplicate name returns 409 Conflict."""
        files = client_from_platform(sdk, FilesClient)
        with create_fileset(sdk) as fileset:
            # Try to create another fileset with the same name and workspace
            try:
                files.create_fileset(
                    body=CreateFilesetRequest(name=fileset.name),
                    workspace=fileset.workspace,
                )
                assert False, "Should have raised ConflictError"
            except ConflictError as e:
                # Verify the error message mentions the conflict
                assert "already exists" in str(e).lower() or e.status_code == 409

    def test_fileset_create_rejects_user_provided_local_storage(self, sdk: NeMoHelix):
        """Test that explicitly requesting local storage is rejected."""
        files = client_from_platform(sdk, FilesClient)
        try:
            files.create_fileset(
                body=CreateFilesetRequest(
                    name="reject-local-storage",
                    storage={"type": "local", "path": "/etc"},
                ),
                workspace=DEFAULT_WORKSPACE_ID,
            )
            assert False, "Should have raised NemoHTTPError for local storage"
        except BadRequestError as exc:
            assert exc.status_code == 400
            assert "local storage is not allowed" in str(exc.body).lower()

    def test_fileset_create_rejects_s3_use_sdk_auth(self, sdk: NeMoHelix):
        """Test that S3 storage with use_sdk_auth=True is rejected for user-provided storage."""
        files = client_from_platform(sdk, FilesClient)
        try:
            files.create_fileset(
                body=CreateFilesetRequest(
                    name="reject-s3-sdk-auth",
                    storage={
                        "type": "s3",
                        "bucket": "my-bucket",
                        "use_sdk_auth": True,
                    },
                ),
                workspace=DEFAULT_WORKSPACE_ID,
            )
            assert False, "Should have raised NemoHTTPError for S3 with use_sdk_auth=True"
        except BadRequestError as exc:
            assert exc.status_code == 400
            assert "use_sdk_auth=true is not allowed" in str(exc.body).lower()

    def test_fileset_create_allows_user_provided_local_storage_when_enabled(
        self, sdk_allow_user_local_storage: NeMoHelix, tmp_path: Path
    ):
        """Test that explicit local storage is allowed when feature flag is enabled."""
        files = client_from_platform(sdk_allow_user_local_storage, FilesClient)
        fileset = files.create_fileset(
            body=CreateFilesetRequest(
                name="allow-local-storage",
                storage={"type": "local", "path": str(tmp_path / "explicit")},
            ),
            workspace=DEFAULT_WORKSPACE_ID,
        ).data()

        assert fileset.storage.type == "local"
        assert fileset.storage.path == str(tmp_path / "explicit")

        # Cleanup because not using create_fileset() helper.
        files.delete_fileset(name=fileset.name, workspace=fileset.workspace)

    def test_fileset_update_partial(self, sdk: NeMoHelix):
        """Test that partial updates work - only specified fields are updated."""
        files = client_from_platform(sdk, FilesClient)
        with create_fileset(
            sdk,
            purpose="generic",
            custom_fields={"key1": "value1", "key2": "value2"},
        ) as fileset:
            original_purpose = fileset.purpose
            original_custom_fields = fileset.custom_fields

            # Update only description
            updated = files.update_fileset(
                name=fileset.name,
                workspace=fileset.workspace,
                body=UpdateFilesetRequest(description="Updated description only"),
            ).data()

            # Verify description was updated
            assert updated.description == "Updated description only"
            # Verify other fields remain unchanged
            assert updated.purpose == original_purpose
            assert updated.custom_fields == original_custom_fields
            assert updated.name == fileset.name
            assert updated.id == fileset.id

    def test_fileset_update_description_purpose_custom_fields(self, sdk: NeMoHelix):
        """Test that description, purpose, and custom_fields can all be updated."""
        files = client_from_platform(sdk, FilesClient)
        with create_fileset(sdk, purpose="generic") as fileset:
            # Update all three fields
            updated = files.update_fileset(
                name=fileset.name,
                workspace=fileset.workspace,
                body=UpdateFilesetRequest(
                    description="New description",
                    purpose="dataset",
                    custom_fields={"new_key": "new_value", "another": 123},
                ),
            ).data()

            # Verify all fields were updated
            assert updated.description == "New description"
            assert updated.purpose == "dataset"
            assert updated.custom_fields == {"new_key": "new_value", "another": 123}

            # Verify by fetching the fileset again
            fetched = files.get_fileset(name=fileset.name, workspace=fileset.workspace).data()
            assert fetched.description == "New description"
            assert fetched.purpose == "dataset"
            assert fetched.custom_fields == {"new_key": "new_value", "another": 123}

    def test_fileset_update_not_found(self, sdk: NeMoHelix):
        """Test that updating a non-existent fileset returns 404."""
        files = client_from_platform(sdk, FilesClient)
        try:
            files.update_fileset(
                name="non-existent-fileset",
                workspace=DEFAULT_WORKSPACE_ID,
                body=UpdateFilesetRequest(description="Should fail"),
            )
            assert False, "Should have raised NotFoundError"
        except NotFoundError:
            pass  # Expected

    def test_fileset_update_returns_updated_output(self, sdk: NeMoHelix):
        """Test that update returns the updated FilesetOutput with correct fields."""
        files = client_from_platform(sdk, FilesClient)
        with create_fileset(sdk, purpose="generic") as fileset:
            updated = files.update_fileset(
                name=fileset.name,
                workspace=fileset.workspace,
                body=UpdateFilesetRequest(
                    description="Updated description",
                    custom_fields={"status": "modified"},
                ),
            ).data()

            # Verify the returned object has all expected fields
            assert updated.id == fileset.id
            assert updated.name == fileset.name
            assert updated.workspace == fileset.workspace
            assert updated.description == "Updated description"
            assert updated.custom_fields == {"status": "modified"}
            assert updated.purpose == fileset.purpose
            assert updated.storage is not None
            assert updated.created_at is not None
            assert updated.updated_at is not None

    def test_fileset_create_with_dataset_metadata(self, sdk: NeMoHelix):
        """Test creating a fileset with dataset purpose and metadata."""
        with create_fileset(
            sdk,
            purpose="dataset",
            metadata={
                "dataset": {
                    "schema_defs": {
                        "default_row": {
                            "type": "object",
                            "properties": {
                                "id": {"type": "string"},
                                "name": {"type": "string"},
                                "value": {"type": "number"},
                            },
                            "required": ["id", "name", "value"],
                        },
                        "validation_row": {
                            "type": "object",
                            "properties": {
                                "id": {"type": "string"},
                                "name": {"type": "string"},
                            },
                            "required": ["id", "name"],
                        },
                    },
                    "schema": "default_row",
                    "schemas_by_path": {
                        "validation.jsonl": "validation_row",
                    },
                }
            },
        ) as fileset:
            assert fileset.purpose == "dataset"
            # SDK uses schema_ to avoid conflict with Python's schema
            assert fileset.metadata.dataset is not None
            assert fileset.metadata.dataset.schema_ == "default_row"
            assert fileset.metadata.dataset.schema_defs == {
                "default_row": {
                    "type": "object",
                    "properties": {
                        "id": {"type": "string"},
                        "name": {"type": "string"},
                        "value": {"type": "number"},
                    },
                    "required": ["id", "name", "value"],
                },
                "validation_row": {
                    "type": "object",
                    "properties": {
                        "id": {"type": "string"},
                        "name": {"type": "string"},
                    },
                    "required": ["id", "name"],
                },
            }
            assert fileset.metadata.dataset.schemas_by_path == {"validation.jsonl": "validation_row"}

    def test_fileset_create_rejects_invalid_dataset_schema_metadata(self, sdk: NeMoHelix):
        """Test invalid JSON Schema metadata is rejected at fileset create time."""
        with pytest.raises(
            (NemoHTTPError, ValidationError),
            match="definitely-not-a-valid-json-schema-type",
        ):
            with create_fileset(
                sdk,
                purpose="dataset",
                metadata={
                    "dataset": {
                        "schema": {"type": "definitely-not-a-valid-json-schema-type"},
                    }
                },
            ):
                pass

    def test_fileset_default_storage_path(self, sdk: NeMoHelix):
        """Test that fileset created without storage uses default with workspace/name subpath."""
        with create_fileset(sdk, purpose="generic") as fileset:
            # Verify storage config was set with the expected subpath
            assert fileset.storage is not None
            assert fileset.storage.type == "local"
            # The path should end with filesets/{workspace}/{name}
            assert fileset.storage.path.endswith(f"filesets/{DEFAULT_WORKSPACE_ID}/{fileset.name}")

    def test_fileset_delete_removes_storage_data(self, sdk: NeMoHelix):
        """Test that deleting a fileset also deletes the underlying storage directory."""
        files = client_from_platform(sdk, FilesClient)
        # Create fileset manually (not using context manager) so we control deletion
        fileset = files.create_fileset(
            body=CreateFilesetRequest(name="delete-storage-test"),
            workspace=DEFAULT_WORKSPACE_ID,
        ).data()

        try:
            # Upload some files
            files.upload_file(
                content=b"content1",
                path="file1.txt",
                name=fileset.name,
                workspace=fileset.workspace,
            )
            files.upload_file(
                content=b"content2",
                path="subdir/file2.txt",
                name=fileset.name,
                workspace=fileset.workspace,
            )

            # Verify storage path exists with files
            assert fileset.storage.type == "local"
            storage_path = Path(fileset.storage.path)
            assert storage_path.exists()
            assert (storage_path / "file1.txt").exists()
            assert (storage_path / "subdir" / "file2.txt").exists()

            # Delete the fileset
            files.delete_fileset(name=fileset.name, workspace=fileset.workspace)

            # Verify storage directory is gone
            assert not storage_path.exists()

        except Exception:
            # Clean up on failure
            try:
                files.delete_fileset(name=fileset.name, workspace=fileset.workspace)
            except Exception:
                pass
            raise

    def test_fileset_list_filter_by_created_at_gte(self, sdk: NeMoHelix):
        """Test listing filesets with created_at[gte] filter."""
        files = client_from_platform(sdk, FilesClient)
        # Record time before creating filesets
        before_create = datetime.now(timezone.utc) - timedelta(seconds=5)

        with create_fileset(sdk) as fileset1:
            time.sleep(1)
            with create_fileset(sdk) as fileset2:
                # Filter by created_at[gte] should include both new filesets
                filtered = list(
                    files.list_filesets(
                        workspace=DEFAULT_WORKSPACE_ID,
                        query_params={"filter": {"created_at": {"$gte": before_create.isoformat(timespec="seconds")}}},
                    ).items()
                )
                fileset_ids = {fs.id for fs in filtered}
                assert fileset1.id in fileset_ids
                assert fileset2.id in fileset_ids

    def test_fileset_list_filter_by_created_at_lte(self, sdk: NeMoHelix):
        """Test listing filesets with created_at[lte] filter."""
        files = client_from_platform(sdk, FilesClient)
        with create_fileset(sdk) as fileset1:
            # Record time after creating first fileset
            after_first = datetime.now(timezone.utc)

            # Delay to ensure second fileset has different timestamp
            time.sleep(1)

            with create_fileset(sdk) as _fileset2:  # noqa: F841
                # Filter by created_at[lte] with time after first fileset
                # should include first fileset but might include second
                # (depends on timing precision)
                filtered = list(
                    files.list_filesets(
                        workspace=DEFAULT_WORKSPACE_ID,
                        query_params={"filter": {"created_at": {"$lte": after_first.isoformat()}}},
                    ).items()
                )
                fileset_ids = {fs.id for fs in filtered}
                assert fileset1.id in fileset_ids

    def test_fileset_list_filter_by_created_at_range(self, sdk: NeMoHelix):
        """Test listing filesets with both created_at[gte] and created_at[lte]."""
        files = client_from_platform(sdk, FilesClient)
        before_create = datetime.now(timezone.utc)

        with create_fileset(sdk) as fileset:
            after_create = datetime.now(timezone.utc)

            # Filter by date range that includes the fileset
            filtered = list(
                files.list_filesets(
                    workspace=DEFAULT_WORKSPACE_ID,
                    query_params={
                        "filter": {
                            "created_at": {
                                "$gte": before_create.isoformat(),
                                "$lte": after_create.isoformat(),
                            }
                        }
                    },
                ).items()
            )
            fileset_ids = {fs.id for fs in filtered}
            assert fileset.id in fileset_ids

    def test_fileset_list_filter_by_created_at_excludes_older(self, sdk: NeMoHelix):
        """Test that created_at[gte] filter excludes older filesets."""
        files = client_from_platform(sdk, FilesClient)
        with create_fileset(sdk) as old_fileset:
            # Delay to ensure timestamps differ
            # Using 1s because SQLite doesn't track sub-second precision
            time.sleep(1)

            # Record time AFTER the sleep to ensure it's in a different second
            # than old_fileset's creation time
            after_old = datetime.now(timezone.utc)

            with create_fileset(sdk) as new_fileset:
                # Filter by created_at[gte] after old fileset was created
                filtered = list(
                    files.list_filesets(
                        workspace=DEFAULT_WORKSPACE_ID,
                        query_params={"filter": {"created_at": {"$gte": after_old.isoformat()}}},
                    ).items()
                )
                fileset_ids = {fs.id for fs in filtered}
                # New fileset should be included
                assert new_fileset.id in fileset_ids
                # Old fileset should be excluded
                assert old_fileset.id not in fileset_ids

    def test_fileset_list_filter_by_updated_at(self, sdk: NeMoHelix):
        """Test listing filesets with updated_at filter."""
        files = client_from_platform(sdk, FilesClient)
        before_create = datetime.now(timezone.utc)

        with create_fileset(sdk) as fileset:
            # Filter by updated_at[gte] should include the fileset
            filtered = list(
                files.list_filesets(
                    workspace=DEFAULT_WORKSPACE_ID,
                    query_params={"filter": {"updated_at": {"$gte": before_create.isoformat()}}},
                ).items()
            )
            fileset_ids = {fs.id for fs in filtered}
            assert fileset.id in fileset_ids

    def test_fileset_list_combined_filters_with_datetime(self, sdk: NeMoHelix):
        """Test combining datetime filters with other filters."""
        files = client_from_platform(sdk, FilesClient)
        before_create = datetime.now(timezone.utc)

        with create_fileset(sdk, purpose="dataset") as dataset_fileset:
            with create_fileset(sdk, purpose="generic") as generic_fileset:
                # Combine purpose filter with created_at filter
                filtered = list(
                    files.list_filesets(
                        workspace=DEFAULT_WORKSPACE_ID,
                        query_params={
                            "filter": {
                                "purpose": "dataset",
                                "created_at": {"$gte": before_create.isoformat()},
                            }
                        },
                    ).items()
                )
                fileset_ids = {fs.id for fs in filtered}
                # Should include dataset fileset
                assert dataset_fileset.id in fileset_ids
                # Should exclude generic fileset
                assert generic_fileset.id not in fileset_ids
