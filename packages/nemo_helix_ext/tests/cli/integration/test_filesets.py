# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Integration tests for filesets CLI commands."""

from pathlib import Path

import pytest
from nemo_helix import NeMoHelix
from nemo_helix_ext.cli.app import app
from nemo_helix_plugin.client.adapter import client_from_platform
from nemo_helix_plugin.files.client import FilesClient
from nemo_helix_plugin.files.types import CreateFilesetRequest

from ..utils import assert_exit_code
from .conftest import NhxCliRunner


@pytest.fixture
def test_fileset(files_client: FilesClient, random_workspace: str) -> dict:
    """Create a test fileset."""
    fileset = files_client.create_fileset(
        body=CreateFilesetRequest(name="test-fileset"), workspace=random_workspace
    ).data()
    return {"workspace": random_workspace, "name": fileset.name}


class TestFilesetsUpload:
    """Tests for filesets upload command."""

    def test_upload_file_basic(
        self,
        runner: NhxCliRunner,
        test_fileset: dict,
        tmp_path: Path,
    ):
        """Test basic file upload to a fileset."""
        # Create a test file
        test_file = tmp_path / "test.txt"
        test_file.write_text("Hello, fileset!")

        # Upload via CLI
        result = runner.invoke(
            app,
            f"files upload {test_file} {test_fileset['name']} --workspace {test_fileset['workspace']} --remote-path test.txt",
        )

        assert_exit_code(result, 0)
        assert "Completed upload to" in result.stdout

        # Verify file exists in fileset
        files_response = runner.client.files.list(
            workspace=test_fileset["workspace"],
            fileset=test_fileset["name"],
        )
        file_paths = [f.path for f in files_response.data]
        assert "test.txt" in file_paths

    def test_upload_dir(
        self,
        runner: NhxCliRunner,
        test_fileset: dict,
        tmp_path: Path,
    ):
        """Test uploading a directory to a fileset.

        Without a trailing slash, the directory itself is copied (creates a subdirectory).
        """
        # Create a directory with files
        subdir = tmp_path / "mydir"
        subdir.mkdir()
        (subdir / "file1.txt").write_text("content1")
        (subdir / "file2.txt").write_text("content2")

        # Upload directory (no trailing slash) - copies the directory itself
        result = runner.invoke(
            app,
            f"files upload {subdir} {test_fileset['name']} --workspace {test_fileset['workspace']}",
        )

        assert_exit_code(result, 0)
        assert "Completed upload to" in result.stdout

        # Verify files exist under mydir/ subdirectory
        files_response = runner.client.files.list(
            workspace=test_fileset["workspace"],
            fileset=test_fileset["name"],
        )
        file_paths = [f.path for f in files_response.data]
        assert "mydir/file1.txt" in file_paths
        assert "mydir/file2.txt" in file_paths

    def test_upload_dir_trailing_slash(
        self,
        runner: NhxCliRunner,
        test_fileset: dict,
        tmp_path: Path,
    ):
        """Test uploading a directory with trailing slash.

        With a trailing slash, the contents of the directory are copied (no subdirectory created).
        See https://filesystem-spec.readthedocs.io/en/latest/copying.html
        """
        # Create a directory with files
        subdir = tmp_path / "mydir"
        subdir.mkdir()
        (subdir / "file1.txt").write_text("content1")
        (subdir / "file2.txt").write_text("content2")

        # Upload directory with trailing slash - copies contents only
        result = runner.invoke(
            app,
            f"files upload {subdir}/ {test_fileset['name']} --workspace {test_fileset['workspace']}",
        )

        assert_exit_code(result, 0)
        assert "Completed upload to" in result.stdout

        # Verify files exist at root level (no mydir/ prefix)
        files_response = runner.client.files.list(
            workspace=test_fileset["workspace"],
            fileset=test_fileset["name"],
        )
        file_paths = [f.path for f in files_response.data]
        assert "file1.txt" in file_paths
        assert "file2.txt" in file_paths

    def test_upload_to_nonexistent_fileset_fails(
        self,
        runner: NhxCliRunner,
        random_workspace: str,
        tmp_path: Path,
    ):
        """Test that uploading to a non-existent fileset gives a clear error."""
        # Create a test file
        test_file = tmp_path / "test.txt"
        test_file.write_text("content")

        # Try to upload to a fileset that doesn't exist
        result = runner.invoke(
            app,
            f"files upload {test_file} nonexistent-fileset-12345 --workspace {random_workspace}",
        )

        assert_exit_code(result, 3)
        assert "not found" in result.stderr.lower()

    @pytest.mark.parametrize(
        ("remote_path", "expected_suffix"),
        [
            ("", ""),  # root upload, no hash
            ("subdir/", "#subdir/"),  # with remote_path, shows hash
        ],
    )
    def test_upload_message_format(
        self,
        runner: NhxCliRunner,
        test_fileset: dict,
        tmp_path: Path,
        remote_path: str,
        expected_suffix: str,
    ):
        """Test that upload completion message shows fileset[#path] format."""
        test_file = tmp_path / "test.txt"
        test_file.write_text("Hello!")

        cmd = f"files upload {test_file} {test_fileset['name']} --workspace {test_fileset['workspace']}"
        if remote_path:
            cmd = f"files upload {test_file} {test_fileset['name']} --workspace {test_fileset['workspace']} --remote-path {remote_path}"

        result = runner.invoke(app, cmd)

        assert_exit_code(result, 0)
        expected = f"Completed upload to {test_fileset['name']}{expected_suffix}"
        assert expected in result.stdout

    def test_upload_without_fileset_auto_creates(
        self,
        runner: NhxCliRunner,
        random_workspace: str,
        tmp_path: Path,
    ):
        """Test that uploads without a fileset name create a new fileset."""
        import re

        test_file = tmp_path / "test.txt"
        test_file.write_text("Hello!")

        result = runner.invoke(
            app,
            f"files upload {test_file} --workspace {random_workspace}",
        )

        assert_exit_code(result, 0)
        assert "Completed upload to" in result.stdout

        # Extract fileset name from message and verify it was created
        # Output format: "Completed upload to fileset-xxxxxxxx"
        match = re.search(r"Completed upload to (fileset-[a-f0-9]+)", result.stdout)
        assert match, f"Should show auto-generated fileset name, got: {result.stdout}"
        fileset_name = match.group(1)

        # Verify fileset exists
        files = client_from_platform(runner.client, FilesClient)
        fileset = files.get_fileset(name=fileset_name, workspace=random_workspace).data()
        assert fileset.name == fileset_name

        # Verify file was uploaded
        files_response = runner.client.files.list(
            workspace=random_workspace,
            fileset=fileset_name,
        )
        file_paths = [f.path for f in files_response.data]
        assert "test.txt" in file_paths


@pytest.fixture
def fileset_with_nested_files(sdk: NeMoHelix, files_client: FilesClient, random_workspace: str, tmp_path: Path) -> dict:
    """Create a fileset with nested file structure for download tests.

    Structure:
        a/
            file1.txt
            b/
                file2.txt
                file3.txt
    """
    fileset = files_client.create_fileset(
        body=CreateFilesetRequest(name="download-test-fileset"), workspace=random_workspace
    ).data()

    # Create nested directory structure locally
    dir_a = tmp_path / "a"
    dir_b = dir_a / "b"
    dir_b.mkdir(parents=True)

    (dir_a / "file1.txt").write_text("content1")
    (dir_b / "file2.txt").write_text("content2")
    (dir_b / "file3.txt").write_text("content3")

    files = client_from_platform(sdk, FilesClient)
    for local in dir_a.rglob("*"):
        if local.is_file():
            files.upload_file(
                name=fileset.name,
                workspace=random_workspace,
                path=f"{dir_a.name}/{local.relative_to(dir_a).as_posix()}",
                content=local.read_bytes(),
            )

    return {"workspace": random_workspace, "name": fileset.name}


class TestFilesetsDownload:
    """Tests for filesets download command."""

    def test_download_single_file(
        self,
        runner: NhxCliRunner,
        fileset_with_nested_files: dict,
        tmp_path: Path,
    ):
        """Test downloading a single file from a fileset."""
        output_dir = tmp_path / "download"
        output_dir.mkdir()

        result = runner.invoke(
            app,
            f"files download {fileset_with_nested_files['name']} --workspace {fileset_with_nested_files['workspace']} --remote-path a/b/file2.txt -o {output_dir}",
        )

        assert_exit_code(result, 0)
        assert "Downloaded" in result.stdout

        # Verify file was downloaded
        downloaded_file = output_dir / "file2.txt"
        assert downloaded_file.exists()
        assert downloaded_file.read_text() == "content2"

    def test_download_one_level(
        self,
        runner: NhxCliRunner,
        fileset_with_nested_files: dict,
        tmp_path: Path,
    ):
        """Test downloading one directory level (a/b/) - should get file2.txt and file3.txt."""
        output_dir = tmp_path / "download"
        output_dir.mkdir()

        result = runner.invoke(
            app,
            f"files download {fileset_with_nested_files['name']} --workspace {fileset_with_nested_files['workspace']} --remote-path a/b/ -o {output_dir}/",
        )

        assert_exit_code(result, 0)
        assert "Downloaded" in result.stdout

        # Verify files from b/ directory were downloaded
        assert (output_dir / "file2.txt").exists()
        assert (output_dir / "file3.txt").exists()
        assert (output_dir / "file2.txt").read_text() == "content2"
        assert (output_dir / "file3.txt").read_text() == "content3"
        # file1.txt should NOT be downloaded (it's in a/, not a/b/)
        assert not (output_dir / "file1.txt").exists()

    def test_download_two_levels(
        self,
        runner: NhxCliRunner,
        fileset_with_nested_files: dict,
        tmp_path: Path,
    ):
        """Test downloading two levels up (a/) - should get all files."""
        output_dir = tmp_path / "download"
        output_dir.mkdir()

        result = runner.invoke(
            app,
            f"files download {fileset_with_nested_files['name']} --workspace {fileset_with_nested_files['workspace']} --remote-path a/ -o {output_dir}/",
        )

        assert_exit_code(result, 0)
        assert "Downloaded" in result.stdout

        # Verify all files were downloaded with their directory structure
        assert (output_dir / "file1.txt").exists()
        assert (output_dir / "b" / "file2.txt").exists()
        assert (output_dir / "b" / "file3.txt").exists()
        assert (output_dir / "file1.txt").read_text() == "content1"
        assert (output_dir / "b" / "file2.txt").read_text() == "content2"
        assert (output_dir / "b" / "file3.txt").read_text() == "content3"
