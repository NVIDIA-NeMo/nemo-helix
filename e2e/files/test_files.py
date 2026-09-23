# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""E2E tests for the files service.

These tests verify basic file upload and download operations
work correctly when running against a fully deployed NHX platform.
"""

import tempfile
import uuid
from collections.abc import Iterator
from pathlib import Path

import pytest
from nemo_helix import NeMoHelix
from nemo_helix_plugin.client.adapter import client_from_platform
from nemo_helix_plugin.files.client import FilesClient
from nemo_helix_plugin.files.types import CreateFilesetRequest, ListFilesQueryParams
from nemo_helix_plugin.files.types import FilesetOutput as Fileset


@pytest.fixture
def fileset(files_client: FilesClient, workspace: str) -> Iterator[Fileset]:
    """Create a unique fileset for each test with automatic cleanup."""
    fileset_name = f"e2e-fileset-{uuid.uuid4().hex[:8]}"
    fileset = files_client.create_fileset(body=CreateFilesetRequest(name=fileset_name), workspace=workspace).data()
    yield fileset
    try:
        files_client.delete_fileset(name=fileset_name, workspace=workspace)
    except Exception:
        pass


def test_file_upload_download(sdk: NeMoHelix, workspace: str, fileset: Fileset):
    """Test uploading and downloading a file.

    This test verifies the files system works end-to-end:
    1. Upload a file with test content
    2. Download the file and verify content matches
    """
    test_content = b"Hello from e2e test! This is test file content."

    # Upload file using high-level API
    files = client_from_platform(sdk, FilesClient)
    files.upload_file(
        name=fileset.name,
        workspace=workspace,
        path="test.txt",
        content=test_content,
    )

    # Verify file was uploaded
    files_list = files.list_files(name=fileset.name, workspace=workspace).data().data
    assert len(files_list) == 1
    assert files_list[0].path == "test.txt"
    assert files_list[0].size == len(test_content)

    # Download file and verify content
    downloaded = files.download_file(
        name=fileset.name,
        workspace=workspace,
        path="test.txt",
    ).read()
    assert downloaded == test_content


def test_file_list_cache_status_for_default_storage(sdk: NeMoHelix, workspace: str, fileset: Fileset):
    """Test cache status reporting for files stored in the default backend."""
    test_content = b"cache status coverage"

    client_from_platform(sdk, FilesClient).upload_file(
        name=fileset.name,
        workspace=workspace,
        path="cache-status.txt",
        content=test_content,
    )

    files = client_from_platform(sdk, FilesClient)
    files_without_cache_check = files.list_files(name=fileset.name, workspace=workspace).data().data
    assert len(files_without_cache_check) == 1
    assert files_without_cache_check[0].cache_status == "not_cacheable"

    files_with_cache_check = (
        files.list_files(
            name=fileset.name,
            workspace=workspace,
            query_params=ListFilesQueryParams(include_cache_status=True),
        )
        .data()
        .data
    )
    assert len(files_with_cache_check) == 1
    assert files_with_cache_check[0].cache_status == "not_cacheable"


def test_file_upload_nested_path(sdk: NeMoHelix, workspace: str, fileset: Fileset):
    """Test uploading a file with a nested path.

    Verifies that files can be uploaded to nested directories
    within a fileset.
    """
    test_content = b"Nested file content"
    test_path = "folder/subfolder/nested.txt"

    # Upload file to nested path
    files = client_from_platform(sdk, FilesClient)
    files.upload_file(
        name=fileset.name,
        workspace=workspace,
        path=test_path,
        content=test_content,
    )

    # List files and verify the nested file appears
    files_list = files.list_files(name=fileset.name, workspace=workspace).data().data
    file_paths = {f.path for f in files_list}
    assert test_path in file_paths

    # Download and verify
    downloaded = files.download_file(
        name=fileset.name,
        workspace=workspace,
        path=test_path,
    ).read()
    assert downloaded == test_content


def test_file_delete(sdk: NeMoHelix, workspace: str, fileset: Fileset):
    """Test deleting a file from a fileset.

    Verifies that files can be deleted and are no longer
    accessible after deletion.
    """
    test_content = b"File to be deleted"
    test_path = "delete-me.txt"

    # Upload file
    files = client_from_platform(sdk, FilesClient)
    files.upload_file(
        name=fileset.name,
        workspace=workspace,
        path=test_path,
        content=test_content,
    )

    # Verify file exists by listing
    files_list = files.list_files(name=fileset.name, workspace=workspace).data().data
    assert any(f.path == test_path for f in files_list)

    # Delete file
    files.delete_file(
        name=fileset.name,
        workspace=workspace,
        path=test_path,
    )

    # Verify file is gone
    files_list = files.list_files(name=fileset.name, workspace=workspace).data().data
    assert not any(f.path == test_path for f in files_list)


def test_directory_upload_and_download(sdk: NeMoHelix, workspace: str, fileset: Fileset):
    """Test uploading and downloading a directory.

    Verifies that directory contents can be uploaded and downloaded
    with their relative structure preserved.
    """
    test_files = {
        "file1.txt": b"content1",
        "file2.txt": b"content2",
        "subdir/file3.txt": b"content3",
    }

    with tempfile.TemporaryDirectory() as tmpdir:
        # Upload each file, preserving its relative path
        files = client_from_platform(sdk, FilesClient)
        for remote_path, content in test_files.items():
            files.upload_file(
                name=fileset.name,
                workspace=workspace,
                path=remote_path,
                content=content,
            )

        # Verify all files were uploaded
        files_list = files.list_files(name=fileset.name, workspace=workspace).data().data
        paths = {f.path for f in files_list}
        assert paths == set(test_files)

        # Download each file, preserving its relative path
        download_root = Path(tmpdir, "download")
        for entry in files_list:
            dest = download_root / entry.path
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(files.download_file(name=fileset.name, workspace=workspace, path=entry.path).read())

        # Verify downloaded content matches
        for remote_path, content in test_files.items():
            assert (download_root / remote_path).read_bytes() == content
