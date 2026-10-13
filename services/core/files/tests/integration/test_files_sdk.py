# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Integration tests for the high-level FilesResource.

These tests verify:
- files_resource.upload() - Upload files/directories
- files_resource.upload_content() - Upload in-memory data
- files_resource.download() - Download files/directories
- files_resource.download_content() - Download file content to memory
- files_resource.list() - List files with FilesetFileOutput objects
- files_resource.delete() - Delete files
- fileset_auto_create parameter for upload operations

Uses the create_test_client pattern for fast in-memory testing.
"""

import json
import tempfile
import uuid
from collections.abc import Callable
from io import BytesIO
from pathlib import Path

import pytest
from filesets.resources import FilesResource
from nemo_helix_plugin.client.client import NemoClient
from nemo_helix_plugin.client.errors import NotFoundError, PermissionDeniedError
from nemo_helix_plugin.files.client import FilesClient
from nemo_helix_plugin.files.types import (
    CreateFilesetRequest,
    FilesetFileOutput,
    FilesetOutput,
    UpdateFilesetRequest,
)
from nhx.core.files.testing.utils import create_fileset, test_fileset_name


class TestFilesUpload:
    """Tests for files_resource.upload()."""

    def test_upload_single_file(self, files_resource: FilesResource, fileset: FilesetOutput, tmp_path: Path):
        """Test uploading a single file."""
        local_file = tmp_path / "upload.txt"
        local_file.write_text("Hello, World!")

        files_resource.upload(
            fileset=fileset.name,
            workspace=fileset.workspace,
            local_path=str(local_file),
            remote_path="test.txt",
        )

        # Verify file was uploaded
        files = files_resource.list(fileset=fileset.name, workspace=fileset.workspace)
        assert len(files.data) == 1
        assert files.data[0].path == "test.txt"
        assert files.data[0].size == len("Hello, World!")

    def test_upload_directory_contents_with_trailing_slash(self, files_resource: FilesResource, fileset: FilesetOutput):
        """Test uploading directory contents (trailing slash on local_path).

        With trailing slash: `upload("mydir/")` copies the CONTENTS of mydir.
        Files end up at the fileset root: file1.txt, subdir/file3.txt
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create directory structure: mydir/file1.txt, mydir/subdir/file3.txt
            mydir = Path(tmpdir, "mydir")
            mydir.mkdir()
            Path(mydir, "file1.txt").write_text("content1")
            subdir = Path(mydir, "subdir")
            subdir.mkdir()
            Path(subdir, "file3.txt").write_text("content3")

            # Upload with trailing slash - copies CONTENTS
            files_resource.upload(
                fileset=fileset.name,
                workspace=fileset.workspace,
                local_path=f"{mydir}/",
                remote_path="",
            )

            # Verify files are at root (not under mydir/)
            files = files_resource.list(fileset=fileset.name, workspace=fileset.workspace)
            paths = {f.path for f in files.data}
            assert "file1.txt" in paths, f"Expected 'file1.txt' in {paths}"
            assert "subdir/file3.txt" in paths, f"Expected 'subdir/file3.txt' in {paths}"
            # Should NOT have mydir/ prefix
            assert not any(p.startswith("mydir/") for p in paths), f"Files should not have 'mydir/' prefix: {paths}"

    def test_upload_directory_itself_without_trailing_slash(
        self, files_resource: FilesResource, fileset: FilesetOutput
    ):
        """Test uploading directory itself (no trailing slash on local_path).

        Without trailing slash: `upload("mydir")` copies the directory ITSELF.
        Files end up under the directory name: mydir/file1.txt, mydir/subdir/file3.txt
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create directory structure: mydir/file1.txt, mydir/subdir/file3.txt
            mydir = Path(tmpdir, "mydir")
            mydir.mkdir()
            Path(mydir, "file1.txt").write_text("content1")
            subdir = Path(mydir, "subdir")
            subdir.mkdir()
            Path(subdir, "file3.txt").write_text("content3")

            # Upload WITHOUT trailing slash - copies the directory ITSELF
            files_resource.upload(
                fileset=fileset.name,
                workspace=fileset.workspace,
                local_path=str(mydir),
                remote_path="",
            )

            # Verify files are under mydir/ prefix
            files = files_resource.list(fileset=fileset.name, workspace=fileset.workspace)
            paths = {f.path for f in files.data}
            assert "mydir/file1.txt" in paths, f"Expected 'mydir/file1.txt' in {paths}"
            assert "mydir/subdir/file3.txt" in paths, f"Expected 'mydir/subdir/file3.txt' in {paths}"
            # Should NOT have files at root
            assert "file1.txt" not in paths, f"'file1.txt' should not be at root: {paths}"

    def test_upload_to_subdirectory(self, files_resource: FilesResource, fileset: FilesetOutput, tmp_path: Path):
        """Test uploading a file to a subdirectory."""
        local_file = tmp_path / "nested.txt"
        local_file.write_text("nested content")

        files_resource.upload(
            fileset=fileset.name,
            workspace=fileset.workspace,
            local_path=str(local_file),
            remote_path="a/b/c/nested.txt",
        )

        files = files_resource.list(fileset=fileset.name, workspace=fileset.workspace)
        assert len(files.data) == 1
        assert files.data[0].path == "a/b/c/nested.txt"


class TestFilesDownload:
    """Tests for files_resource.download()."""

    def test_download_single_file(self, files_resource: FilesResource, fileset: FilesetOutput):
        """Test downloading a single file."""
        # First upload a file
        test_content = b"Download test content"
        files_resource.upload_content(
            content=test_content,
            remote_path="test.txt",
            fileset=fileset.name,
            workspace=fileset.workspace,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            files_resource.download(
                fileset=fileset.name,
                workspace=fileset.workspace,
                remote_path="test.txt",
                local_path=f"{tmpdir}/downloaded.txt",
            )

            downloaded = Path(tmpdir, "downloaded.txt").read_bytes()
            assert downloaded == test_content

    def test_download_directory(self, files_resource: FilesResource, fileset: FilesetOutput):
        """Test downloading an entire directory."""
        # Upload multiple files
        files_resource.upload_content(
            content=b"content1",
            remote_path="data/file1.txt",
            fileset=fileset.name,
            workspace=fileset.workspace,
        )
        files_resource.upload_content(
            content=b"content2",
            remote_path="data/file2.txt",
            fileset=fileset.name,
            workspace=fileset.workspace,
        )
        files_resource.upload_content(
            content=b"content3",
            remote_path="data/nested/file3.txt",
            fileset=fileset.name,
            workspace=fileset.workspace,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            files_resource.download(
                fileset=fileset.name,
                workspace=fileset.workspace,
                remote_path="data/",
                local_path=f"{tmpdir}/",
            )

            # Verify all files were downloaded
            assert Path(tmpdir, "file1.txt").read_bytes() == b"content1"
            assert Path(tmpdir, "file2.txt").read_bytes() == b"content2"
            assert Path(tmpdir, "nested/file3.txt").read_bytes() == b"content3"

    def test_download_entire_fileset(self, files_resource: FilesResource, fileset: FilesetOutput):
        """Test downloading all files from a fileset using default remote_path.

        Downloading a fileset copies contents directly. Users who want a subfolder
        can include the fileset name in local_path.
        """
        # Upload files at root
        files_resource.upload_content(
            content=b"root content",
            remote_path="root.txt",
            fileset=fileset.name,
            workspace=fileset.workspace,
        )
        files_resource.upload_content(
            content=b"nested content",
            remote_path="subdir/nested.txt",
            fileset=fileset.name,
            workspace=fileset.workspace,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            # Download everything (remote_path defaults to "")
            # Contents are copied directly to local_path
            files_resource.download(
                fileset=fileset.name,
                workspace=fileset.workspace,
                local_path=f"{tmpdir}/",
            )

            # Files are directly in tmpdir/ (no fileset subfolder)
            assert Path(tmpdir, "root.txt").read_bytes() == b"root content"
            assert Path(tmpdir, "subdir/nested.txt").read_bytes() == b"nested content"
            assert not Path(tmpdir, fileset.name).exists()


class TestFilesList:
    """Tests for files_resource.list()."""

    def test_list_empty_fileset(self, files_resource: FilesResource, fileset: FilesetOutput):
        """Test listing files in an empty fileset."""
        files = files_resource.list(fileset=fileset.name, workspace=fileset.workspace)
        assert files.data == []

    def test_list_returns_fileset_file_objects(self, files_resource: FilesResource, fileset: FilesetOutput):
        """Test that list returns FilesetFileOutput objects with correct attributes."""
        content = b"test content for size check"
        files_resource.upload_content(
            content=content,
            remote_path="test.txt",
            fileset=fileset.name,
            workspace=fileset.workspace,
        )

        files = files_resource.list(fileset=fileset.name, workspace=fileset.workspace)
        assert len(files.data) == 1

        file = files.data[0]
        assert file.path == "test.txt"
        assert file.size == len(content)
        assert file.file_ref == f"{fileset.workspace}/{fileset.name}#test.txt"
        assert file.file_url == f"/apis/files/v2/workspaces/{fileset.workspace}/filesets/{fileset.name}/-/test.txt"

    def test_list_multiple_files(self, files_resource: FilesResource, fileset: FilesetOutput):
        """Test listing multiple files."""
        files_resource.upload_content(
            content=b"a",
            remote_path="file1.txt",
            fileset=fileset.name,
            workspace=fileset.workspace,
        )
        files_resource.upload_content(
            content=b"bb",
            remote_path="file2.txt",
            fileset=fileset.name,
            workspace=fileset.workspace,
        )
        files_resource.upload_content(
            content=b"ccc",
            remote_path="dir/file3.txt",
            fileset=fileset.name,
            workspace=fileset.workspace,
        )

        files = files_resource.list(fileset=fileset.name, workspace=fileset.workspace)
        assert len(files.data) == 3

        paths = {f.path for f in files.data}
        assert paths == {"file1.txt", "file2.txt", "dir/file3.txt"}

        sizes = {f.path: f.size for f in files.data}
        assert sizes["file1.txt"] == 1
        assert sizes["file2.txt"] == 2
        assert sizes["dir/file3.txt"] == 3

    def test_list_subdirectory(self, files_resource: FilesResource, fileset: FilesetOutput):
        """Test listing files in a subdirectory."""
        files_resource.upload_content(
            content=b"root",
            remote_path="root.txt",
            fileset=fileset.name,
            workspace=fileset.workspace,
        )
        files_resource.upload_content(
            content=b"data1",
            remote_path="data/file1.txt",
            fileset=fileset.name,
            workspace=fileset.workspace,
        )
        files_resource.upload_content(
            content=b"data2",
            remote_path="data/file2.txt",
            fileset=fileset.name,
            workspace=fileset.workspace,
        )
        files_resource.upload_content(
            content=b"other",
            remote_path="other/file.txt",
            fileset=fileset.name,
            workspace=fileset.workspace,
        )

        # List only data/ directory
        files = files_resource.list(
            fileset=fileset.name,
            workspace=fileset.workspace,
            remote_path="data/",
        )

        paths = {f.path for f in files.data}
        assert paths == {"data/file1.txt", "data/file2.txt"}

    def test_list_with_path_format(self, files_resource: FilesResource, fileset: FilesetOutput):
        """Test listing using full path format instead of explicit fileset param."""
        files_resource.upload_content(
            content=b"content",
            remote_path="test.txt",
            fileset=fileset.name,
            workspace=fileset.workspace,
        )

        # Use the new path format: workspace/fileset#path
        files = files_resource.list(
            remote_path=f"{fileset.workspace}/{fileset.name}#",
        )

        assert len(files.data) == 1
        assert files.data[0].path == "test.txt"

    def test_list_with_glob_pattern(self, files_resource: FilesResource, fileset: FilesetOutput):
        """Test listing files matching a glob pattern."""
        files_resource.upload_content(
            content=b"json",
            remote_path="data.json",
            fileset=fileset.name,
            workspace=fileset.workspace,
        )
        files_resource.upload_content(
            content=b"config",
            remote_path="config.json",
            fileset=fileset.name,
            workspace=fileset.workspace,
        )
        files_resource.upload_content(
            content=b"readme",
            remote_path="readme.txt",
            fileset=fileset.name,
            workspace=fileset.workspace,
        )
        files_resource.upload_content(
            content=b"nested",
            remote_path="subdir/nested.json",
            fileset=fileset.name,
            workspace=fileset.workspace,
        )

        # List only .json files at root level
        files = files_resource.list(
            fileset=fileset.name,
            workspace=fileset.workspace,
            remote_path="*.json",
        )

        paths = {f.path for f in files.data}
        assert paths == {"data.json", "config.json"}

    def test_list_with_glob_pattern_in_subdirectory(self, files_resource: FilesResource, fileset: FilesetOutput):
        """Test listing files matching a glob pattern in a subdirectory."""
        files_resource.upload_content(
            content=b"train",
            remote_path="data/train.jsonl",
            fileset=fileset.name,
            workspace=fileset.workspace,
        )
        files_resource.upload_content(
            content=b"val",
            remote_path="data/val.jsonl",
            fileset=fileset.name,
            workspace=fileset.workspace,
        )
        files_resource.upload_content(
            content=b"yaml",
            remote_path="data/config.yaml",
            fileset=fileset.name,
            workspace=fileset.workspace,
        )
        files_resource.upload_content(
            content=b"other",
            remote_path="other/file.jsonl",
            fileset=fileset.name,
            workspace=fileset.workspace,
        )

        # List only .jsonl files in data/ directory
        files = files_resource.list(
            fileset=fileset.name,
            workspace=fileset.workspace,
            remote_path="data/*.jsonl",
        )

        paths = {f.path for f in files.data}
        assert paths == {"data/train.jsonl", "data/val.jsonl"}


class TestFilesGlobDownload:
    """Tests for files_resource.download() with glob patterns."""

    def test_download_with_glob_pattern(self, files_resource: FilesResource, fileset: FilesetOutput, tmp_path):
        """Test downloading files matching a glob pattern."""
        files_resource.upload_content(
            content=b"json content",
            remote_path="data.json",
            fileset=fileset.name,
            workspace=fileset.workspace,
        )
        files_resource.upload_content(
            content=b"config content",
            remote_path="config.json",
            fileset=fileset.name,
            workspace=fileset.workspace,
        )
        files_resource.upload_content(
            content=b"readme content",
            remote_path="readme.txt",
            fileset=fileset.name,
            workspace=fileset.workspace,
        )

        # Download only .json files
        files_resource.download(
            fileset=fileset.name,
            workspace=fileset.workspace,
            remote_path="*.json",
            local_path=str(tmp_path),
        )

        # Check only json files were downloaded
        downloaded = list(tmp_path.rglob("*"))
        downloaded_names = {f.name for f in downloaded if f.is_file()}
        assert downloaded_names == {"data.json", "config.json"}

        # Verify content
        assert (tmp_path / "data.json").read_bytes() == b"json content"
        assert (tmp_path / "config.json").read_bytes() == b"config content"

    def test_download_with_glob_pattern_preserves_structure(
        self, files_resource: FilesResource, fileset: FilesetOutput, tmp_path
    ):
        """Test that downloading with glob pattern preserves directory structure."""
        files_resource.upload_content(
            content=b"train data",
            remote_path="data/train.jsonl",
            fileset=fileset.name,
            workspace=fileset.workspace,
        )
        files_resource.upload_content(
            content=b"val data",
            remote_path="data/val.jsonl",
            fileset=fileset.name,
            workspace=fileset.workspace,
        )
        files_resource.upload_content(
            content=b"yaml",
            remote_path="data/config.yaml",
            fileset=fileset.name,
            workspace=fileset.workspace,
        )

        # Download only .jsonl files from data/
        files_resource.download(
            fileset=fileset.name,
            workspace=fileset.workspace,
            remote_path="data/*.jsonl",
            local_path=str(tmp_path),
        )

        # Check directory structure is preserved
        assert (tmp_path / "data" / "train.jsonl").exists()
        assert (tmp_path / "data" / "val.jsonl").exists()
        assert not (tmp_path / "data" / "config.yaml").exists()

        # Verify content
        assert (tmp_path / "data" / "train.jsonl").read_bytes() == b"train data"
        assert (tmp_path / "data" / "val.jsonl").read_bytes() == b"val data"


class TestFilesDelete:
    """Tests for files_resource.delete()."""

    def test_delete_single_file(self, files_resource: FilesResource, fileset: FilesetOutput):
        """Test deleting a single file."""
        files_resource.upload_content(
            content=b"delete me",
            remote_path="to_delete.txt",
            fileset=fileset.name,
            workspace=fileset.workspace,
        )

        # Verify file exists
        files = files_resource.list(fileset=fileset.name, workspace=fileset.workspace)
        assert len(files.data) == 1

        # Delete the file
        files_resource.delete(
            fileset=fileset.name,
            workspace=fileset.workspace,
            remote_path="to_delete.txt",
        )

        # Verify file was deleted
        files = files_resource.list(fileset=fileset.name, workspace=fileset.workspace)
        assert len(files.data) == 0

    def test_delete_nested_file(self, files_resource: FilesResource, fileset: FilesetOutput):
        """Test deleting a file in a nested directory."""
        files_resource.upload_content(
            content=b"nested",
            remote_path="a/b/c/nested.txt",
            fileset=fileset.name,
            workspace=fileset.workspace,
        )

        files_resource.delete(
            fileset=fileset.name,
            workspace=fileset.workspace,
            remote_path="a/b/c/nested.txt",
        )

        files = files_resource.list(fileset=fileset.name, workspace=fileset.workspace)
        assert len(files.data) == 0

    def test_delete_with_path_format(self, files_resource: FilesResource, fileset: FilesetOutput):
        """Test deleting using full path format."""
        files_resource.upload_content(
            content=b"content",
            remote_path="test.txt",
            fileset=fileset.name,
            workspace=fileset.workspace,
        )

        # Delete using the new path format
        files_resource.delete(
            remote_path=f"{fileset.workspace}/{fileset.name}#test.txt",
        )

        files = files_resource.list(fileset=fileset.name, workspace=fileset.workspace)
        assert len(files.data) == 0


class TestFilesRoundTrip:
    """End-to-end tests combining multiple operations."""

    def test_upload_list_download_delete_cycle(self, files_resource: FilesResource, fileset: FilesetOutput):
        """Test a complete cycle of file operations."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create local files
            local_dir = Path(tmpdir, "upload")
            local_dir.mkdir()
            Path(local_dir, "data.json").write_text('{"key": "value"}')
            Path(local_dir, "config.yaml").write_text("setting: true")

            # Upload
            files_resource.upload(
                fileset=fileset.name,
                workspace=fileset.workspace,
                local_path=f"{local_dir}/",
                remote_path="",
            )

            # List and verify
            files = files_resource.list(fileset=fileset.name, workspace=fileset.workspace)
            assert len(files.data) == 2
            paths = {f.path for f in files.data}
            assert paths == {"data.json", "config.yaml"}

            # Download to different location
            # Contents are copied directly
            download_dir = Path(tmpdir, "download")
            download_dir.mkdir()
            files_resource.download(
                fileset=fileset.name,
                workspace=fileset.workspace,
                local_path=f"{download_dir}/",
            )

            # Verify downloaded content matches (directly in download_dir)
            assert Path(download_dir, "data.json").read_text() == '{"key": "value"}'
            assert Path(download_dir, "config.yaml").read_text() == "setting: true"
            assert not (download_dir / fileset.name).exists()

            # Delete one file
            files_resource.delete(
                fileset=fileset.name,
                workspace=fileset.workspace,
                remote_path="data.json",
            )

            # Verify only one file remains
            files = files_resource.list(fileset=fileset.name, workspace=fileset.workspace)
            assert len(files.data) == 1
            assert files.data[0].path == "config.yaml"

    def test_large_directory_upload_download(self, client: NemoClient, files_resource: FilesResource):
        """Test uploading and downloading a larger directory structure."""
        with create_fileset(FilesClient.from_client(client)) as fileset:
            with tempfile.TemporaryDirectory() as tmpdir:
                # Create a directory with multiple files
                upload_dir = Path(tmpdir, "upload")
                upload_dir.mkdir()

                file_count = 10
                for i in range(file_count):
                    subdir = upload_dir / f"dir{i % 3}"
                    subdir.mkdir(exist_ok=True)
                    (subdir / f"file{i}.txt").write_text(f"content {i}")

                # Upload
                files_resource.upload(
                    fileset=fileset.name,
                    workspace=fileset.workspace,
                    local_path=f"{upload_dir}/",
                    remote_path="",
                )

                # List and verify count
                files = files_resource.list(fileset=fileset.name, workspace=fileset.workspace)
                assert len(files.data) == file_count

                # Download (contents copied directly)
                download_dir = Path(tmpdir, "download")
                download_dir.mkdir()
                files_resource.download(
                    fileset=fileset.name,
                    workspace=fileset.workspace,
                    local_path=f"{download_dir}/",
                )

                # Verify all files downloaded correctly (directly in download_dir)
                for i in range(file_count):
                    downloaded = (download_dir / f"dir{i % 3}" / f"file{i}.txt").read_text()
                    assert downloaded == f"content {i}"
                assert not (download_dir / fileset.name).exists()


def _chunk_generator():
    yield b"chunk1"
    yield b"chunk2"
    yield b"chunk3"


class TestFilesUploadContent:
    """Tests for files_resource.upload_content()."""

    @pytest.mark.parametrize(
        ("content", "expected_bytes"),
        [
            pytest.param(b"Hello, World!", b"Hello, World!", id="bytes"),
            pytest.param(
                "Hello, Unicode! 你好",
                "Hello, Unicode! 你好".encode("utf-8"),
                id="string",
            ),
            pytest.param(BytesIO(b"BytesIO content"), b"BytesIO content", id="bytesio"),
            pytest.param(_chunk_generator(), b"chunk1chunk2chunk3", id="iterator"),
        ],
    )
    def test_upload_content(
        self, files_resource: FilesResource, fileset: FilesetOutput, content, expected_bytes: bytes
    ):
        """Test uploading different content types."""
        result = files_resource.upload_content(
            content=content,
            remote_path="test.bin",
            fileset=fileset.name,
            workspace=fileset.workspace,
        )

        assert isinstance(result, FilesetOutput)
        assert result.name == fileset.name
        assert result.workspace == fileset.workspace

        downloaded = files_resource.download_content(
            remote_path="test.bin",
            fileset=fileset.name,
            workspace=fileset.workspace,
        )
        assert downloaded == expected_bytes

    def test_upload_content_to_subdirectory(self, files_resource: FilesResource, fileset: FilesetOutput):
        """Test uploading data to a nested path."""
        files_resource.upload_content(
            content=b"nested content",
            remote_path="a/b/c/nested.txt",
            fileset=fileset.name,
            workspace=fileset.workspace,
        )

        files = files_resource.list(fileset=fileset.name, workspace=fileset.workspace)
        assert len(files.data) == 1
        assert files.data[0].path == "a/b/c/nested.txt"


class TestFilesDownloadContent:
    """Tests for files_resource.download_content()."""

    @pytest.mark.parametrize(
        ("upload_content", "expected_bytes"),
        [
            pytest.param(b"Hello, World!", b"Hello, World!", id="bytes"),
            pytest.param(
                "Hello, Unicode! 你好",
                "Hello, Unicode! 你好".encode("utf-8"),
                id="string_utf8",
            ),
            pytest.param(
                json.dumps({"key": "value", "number": 42}),
                b'{"key": "value", "number": 42}',
                id="json",
            ),
        ],
    )
    def test_download_content(
        self, files_resource: FilesResource, fileset: FilesetOutput, upload_content, expected_bytes: bytes
    ):
        """Test download_content returns correct bytes for different content types."""
        files_resource.upload_content(
            content=upload_content,
            remote_path="test.bin",
            fileset=fileset.name,
            workspace=fileset.workspace,
        )

        result = files_resource.download_content(
            remote_path="test.bin",
            fileset=fileset.name,
            workspace=fileset.workspace,
        )

        assert isinstance(result, bytes)
        assert result == expected_bytes

    def test_download_content_with_path_format(self, files_resource: FilesResource, fileset: FilesetOutput):
        """Test download_content using full path format."""
        files_resource.upload_content(
            content=b"content",
            remote_path="test.txt",
            fileset=fileset.name,
            workspace=fileset.workspace,
        )

        # Use full path format
        downloaded = files_resource.download_content(
            remote_path=f"{fileset.workspace}/{fileset.name}#test.txt",
        )
        assert downloaded == b"content"


class TestFilesUploadAutoCreate:
    """Tests for fileset_auto_create parameter."""

    def test_upload_creates_fileset(
        self, client: NemoClient, files_resource: FilesResource, tmp_path: Path, fileset_cleanup: Callable[[str], None]
    ):
        """Test that upload() with fileset_auto_create creates the fileset."""
        fileset_name = f"auto-create-upload-{uuid.uuid4().hex[:8]}"
        workspace = client.workspace or "default"
        fileset_cleanup(fileset_name)

        local_file = tmp_path / "test.txt"
        local_file.write_text("test content")

        result = files_resource.upload(
            local_path=str(local_file),
            remote_path="test.txt",
            fileset=fileset_name,
            workspace=workspace,
            fileset_auto_create=True,
        )

        # Verify return type is FilesetOutput with correct info
        assert isinstance(result, FilesetOutput)
        assert result.name == fileset_name
        assert result.workspace == workspace

        # Verify fileset was created and file uploaded
        files = files_resource.list(fileset=fileset_name, workspace=workspace)
        assert len(files.data) == 1
        assert files.data[0].path == "test.txt"

    def test_upload_content_creates_fileset(
        self, client: NemoClient, files_resource: FilesResource, fileset_cleanup: Callable[[str], None]
    ):
        """Test that upload_content() with fileset_auto_create creates the fileset."""
        fileset_name = f"auto-create-data-{uuid.uuid4().hex[:8]}"
        workspace = client.workspace or "default"
        fileset_cleanup(fileset_name)

        result = files_resource.upload_content(
            content=b"test content",
            remote_path="test.txt",
            fileset=fileset_name,
            workspace=workspace,
            fileset_auto_create=True,
        )

        # Verify return type is FilesetOutput with correct info
        assert isinstance(result, FilesetOutput)
        assert result.name == fileset_name
        assert result.workspace == workspace

        # Verify fileset was created and file uploaded
        files = files_resource.list(fileset=fileset_name, workspace=workspace)
        assert len(files.data) == 1
        assert files.data[0].path == "test.txt"

    def test_upload_without_flag_fails_for_nonexistent_fileset(self, client: NemoClient, files_resource: FilesResource):
        """Test that upload without flag fails for non-existent fileset."""
        fileset_name = f"nonexistent-{uuid.uuid4().hex[:8]}"
        workspace = client.workspace or "default"

        with pytest.raises(NotFoundError):
            files_resource.upload_content(
                content=b"test",
                remote_path="test.txt",
                fileset=fileset_name,
                workspace=workspace,
                fileset_auto_create=False,
            )

    def test_existing_fileset_with_flag_succeeds(self, files_resource: FilesResource, fileset: FilesetOutput):
        """Test that fileset_auto_create works for existing filesets."""
        result = files_resource.upload_content(
            content=b"test content",
            remote_path="test.txt",
            fileset=fileset.name,
            workspace=fileset.workspace,
            fileset_auto_create=True,  # Should not fail even though fileset exists
        )

        assert isinstance(result, FilesetOutput)
        assert result.name == fileset.name

        files = files_resource.list(fileset=fileset.name, workspace=fileset.workspace)
        assert len(files.data) == 1

    def test_upload_returns_fileset(self, files_resource: FilesResource, fileset: FilesetOutput, tmp_path: Path):
        """Test that upload() always returns the FilesetOutput entity."""
        local_file = tmp_path / "test.txt"
        local_file.write_text("content")

        result = files_resource.upload(
            local_path=str(local_file),
            remote_path="test.txt",
            fileset=fileset.name,
            workspace=fileset.workspace,
        )

        # Even without fileset_auto_create, upload now returns FilesetOutput
        assert isinstance(result, FilesetOutput)
        assert result.name == fileset.name
        assert result.workspace == fileset.workspace

    def test_auto_create_generates_name_when_no_fileset_specified(
        self, client: NemoClient, files_resource: FilesResource, fileset_cleanup: Callable[[str], None]
    ):
        """Test that fileset_auto_create generates a UUID-based name when no fileset is specified."""
        workspace = client.workspace or "default"

        result = files_resource.upload_content(
            content=b"auto-generated fileset test",
            remote_path="test.txt",
            fileset_auto_create=True,
            workspace=workspace,
            # Note: no fileset= parameter
        )

        # Register for cleanup now that we know the name
        fileset_cleanup(result.name)

        # Should return a FilesetOutput with a generated name
        assert isinstance(result, FilesetOutput)
        assert result.name.startswith("fileset-")
        assert len(result.name) == len("fileset-") + 8  # "fileset-" + 8 hex chars

        # Verify file was uploaded
        files = files_resource.list(fileset=result.name, workspace=workspace)
        assert len(files.data) == 1
        assert files.data[0].path == "test.txt"

    def test_auto_create_uses_fileset_from_path_syntax(
        self, client: NemoClient, files_resource: FilesResource, fileset_cleanup: Callable[[str], None]
    ):
        """Test that fileset_auto_create uses fileset from path when # syntax is used."""
        fileset_name = f"path-syntax-{uuid.uuid4().hex[:8]}"
        workspace = client.workspace or "default"
        fileset_cleanup(fileset_name)

        # Use the # syntax to embed fileset in path
        result = files_resource.upload_content(
            content=b"path syntax test",
            remote_path=f"{fileset_name}#data/test.txt",
            fileset_auto_create=True,
            workspace=workspace,
            # Note: no fileset= parameter, but fileset is in path
        )

        # Should use the fileset from the path, not generate a new one
        assert isinstance(result, FilesetOutput)
        assert result.name == fileset_name  # Should NOT be "fileset-..."

        # Verify file was uploaded to correct path
        files = files_resource.list(fileset=fileset_name, workspace=workspace)
        assert len(files.data) == 1
        assert files.data[0].path == "data/test.txt"


class TestListFilesResponseCacheStatus:
    """Tests for ListFilesResponse.cache_status aggregation property."""

    @pytest.mark.parametrize(
        ("statuses", "expected"),
        [
            pytest.param([], None, id="empty_list"),
            pytest.param([None, None], None, id="all_none"),
            pytest.param(["cached", "cached"], "cached", id="all_cached"),
            pytest.param(
                ["cached", "caching", "not_cached"],
                "caching",
                id="caching_takes_priority",
            ),
            pytest.param(
                ["cached", "not_cached"],
                "not_cached",
                id="not_cached_over_cached",
            ),
            pytest.param(
                ["not_cacheable", "not_cacheable"],
                "not_cacheable",
                id="all_not_cacheable",
            ),
            pytest.param(
                ["cached", "not_cacheable"],
                "cached",
                id="mixed_cached_and_not_cacheable",
            ),
        ],
    )
    def test_cache_status_aggregation(self, statuses: list, expected: str | None):
        """Test cache_status aggregation logic.

        Priority order: caching > not_cached > cached > not_cacheable
        Returns None for empty list or when all files have None status.
        """
        from filesets import ListFilesResponse

        files = [
            FilesetFileOutput(
                path=f"file{i}.txt",
                size=100 * (i + 1),
                file_ref=f"ws/fs#file{i}.txt",
                file_url=f"/file{i}.txt",
                cache_status=status,
            )
            for i, status in enumerate(statuses)
        ]
        response = ListFilesResponse(data=files)
        assert response.cache_status == expected


class TestFilesListCacheStatus:
    """Tests for files_resource.list() with include_cache_status parameter."""

    def test_list_with_include_cache_status(self, files_resource: FilesResource, fileset: FilesetOutput):
        """Test listing files with cache status included."""
        files_resource.upload_content(
            content=b"test content",
            remote_path="test.txt",
            fileset=fileset.name,
            workspace=fileset.workspace,
        )

        # List with cache status
        files = files_resource.list(
            fileset=fileset.name,
            workspace=fileset.workspace,
            include_cache_status=True,
        )

        assert len(files.data) == 1
        # For local storage, cache_status is typically None or "not_cacheable"
        # The important thing is that the parameter is passed through correctly
        assert files.data[0].path == "test.txt"

    def test_list_without_include_cache_status(self, files_resource: FilesResource, fileset: FilesetOutput):
        """Test listing files without cache status (default)."""
        files_resource.upload_content(
            content=b"test content",
            remote_path="test.txt",
            fileset=fileset.name,
            workspace=fileset.workspace,
        )

        # List without cache status (default)
        files = files_resource.list(
            fileset=fileset.name,
            workspace=fileset.workspace,
        )

        assert len(files.data) == 1
        assert files.data[0].path == "test.txt"


class TestFilesDownloadEdgeCases:
    """Tests for files_resource.download() edge cases."""

    def test_download_glob_no_matches(self, files_resource: FilesResource, fileset: FilesetOutput, tmp_path):
        """Test downloading with glob pattern that matches no files."""
        # Upload a file that won't match the pattern
        files_resource.upload_content(
            content=b"content",
            remote_path="data.txt",
            fileset=fileset.name,
            workspace=fileset.workspace,
        )

        # Download with glob that matches nothing
        files_resource.download(
            fileset=fileset.name,
            workspace=fileset.workspace,
            remote_path="*.json",  # No .json files exist
            local_path=str(tmp_path),
        )

        # Should complete without error, no files downloaded
        downloaded = list(tmp_path.rglob("*"))
        assert len([f for f in downloaded if f.is_file()]) == 0

    def test_download_content_non_existent_file(self, files_resource: FilesResource, fileset: FilesetOutput):
        """Test downloading content of a file that doesn't exist raises NotFoundError."""
        # Binary/streaming errors are deferred (raised after send()), bypassing remapping.
        with pytest.raises(NotFoundError):
            files_resource.download_content(
                fileset=fileset.name,
                workspace=fileset.workspace,
                remote_path="non-existent.txt",
            )


class TestFilesDeleteEdgeCases:
    """Tests for files_resource.delete() edge cases."""

    def test_delete_non_existent_file(self, files_resource: FilesResource, fileset: FilesetOutput):
        """Test deleting a file that doesn't exist raises NotFoundError."""
        # File delete goes through fsspec rm → deferred error path.
        with pytest.raises(NotFoundError):
            files_resource.delete(
                fileset=fileset.name,
                workspace=fileset.workspace,
                remote_path="non-existent.txt",
            )


class TestFilesetImmutabilityForNonServicePrincipals:
    """Test service_source immutability: only service principals can set/change it; uploads are restricted."""

    def test_create_fileset_with_service_source_as_default_principal_fails_to_set(self, client: NemoClient):
        """Non-service principal cannot set service_source; it is stripped on create."""
        files = FilesClient.from_client(client)
        workspace = client.workspace or "default"
        name = test_fileset_name()
        files.create_fileset(
            body=CreateFilesetRequest(
                name=name,
                description="Test",
                custom_fields={"service_source": "customizer"},
            ),
            workspace=workspace,
        )
        created = files.get_fileset(name=name, workspace=workspace).data()
        # Endpoint strips service_source for non-service principals; fileset must not have it.
        assert created.custom_fields.get("service_source") is None
        files.delete_fileset(name=name, workspace=workspace)

    def test_service_principal_can_set_service_source_and_upload_then_user_cannot_upload(
        self, client_user_and_service: tuple[NemoClient, NemoClient]
    ):
        """service:customizer can create with service_source and upload; non-service principal cannot upload."""
        client_user, client_service = client_user_and_service
        workspace = client_service.workspace or "default"
        name = test_fileset_name()
        # Service principal creates fileset with service_source and uploads a file.
        files = FilesClient.from_client(client_service)
        created = files.create_fileset(
            workspace=workspace,
            body=CreateFilesetRequest(
                name=name,
                description="Immutability test",
                custom_fields={"service_source": "customizer"},
            ),
        ).data()
        assert created.custom_fields.get("service_source") == "customizer"
        files.upload_file(
            content=b"from service",
            path="data.txt",
            name=name,
            workspace=workspace,
        )
        file_list = FilesResource(client_service).list(fileset=name, workspace=workspace)
        assert len(file_list.data) == 1
        assert file_list.data[0].path == "data.txt"
        # Non-service principal must not be able to upload (fileset is immutable for them).
        with pytest.raises(PermissionDeniedError):
            FilesClient.from_client(client_user).upload_file(
                content=b"from user",
                path="user.txt",
                name=name,
                workspace=workspace,
            )
        files.delete_fileset(name=name, workspace=workspace)

    def test_non_service_principal_cannot_overwrite_or_remove_service_source_on_update(
        self, client_user_and_service: tuple[NemoClient, NemoClient]
    ):
        """Non-service principal cannot overwrite, change, or remove service_source via PATCH."""
        client_user, client_service = client_user_and_service
        workspace = client_service.workspace or "default"
        name = test_fileset_name()
        # Service principal creates fileset with service_source.
        service_files = FilesClient.from_client(client_service)
        user_files = FilesClient.from_client(client_user)
        service_files.create_fileset(
            workspace=workspace,
            body=CreateFilesetRequest(
                name=name,
                description="Update immutability test",
                custom_fields={"service_source": "customizer"},
            ),
        )
        user_files.update_fileset(
            name=name,
            workspace=workspace,
            body=UpdateFilesetRequest(custom_fields={"service_source": "other-service"}),
        )
        updated = user_files.get_fileset(name=name, workspace=workspace).data()
        assert updated.custom_fields.get("service_source") == "customizer"
        user_files.update_fileset(
            name=name,
            workspace=workspace,
            body=UpdateFilesetRequest(custom_fields={"other_key": "value"}),
        )
        after_remove_attempt = user_files.get_fileset(name=name, workspace=workspace).data()
        assert after_remove_attempt.custom_fields.get("service_source") == "customizer"
        service_files.delete_fileset(name=name, workspace=workspace)
