# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Integration tests for ``nemo files`` against the in-process Files service."""

import json
import re
import uuid
from pathlib import Path

import pytest
from nemo_platform_ext.cli.app import app
from nemo_platform_plugin.files.client import FilesClient
from nemo_platform_plugin.files.types import CreateFilesetRequest

from ..utils import assert_exit_code
from .conftest import NmpCliRunner


def _list_paths(files_client: FilesClient, workspace: str, fileset: str) -> list[str]:
    response = files_client.list_files(workspace=workspace, name=fileset).data()
    return [f.path for f in response.data]


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
        runner: NmpCliRunner,
        files_client: FilesClient,
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
        assert "test.txt" in _list_paths(files_client, test_fileset["workspace"], test_fileset["name"])

    def test_upload_dir(
        self,
        runner: NmpCliRunner,
        files_client: FilesClient,
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
        file_paths = _list_paths(files_client, test_fileset["workspace"], test_fileset["name"])
        assert "mydir/file1.txt" in file_paths
        assert "mydir/file2.txt" in file_paths

    def test_upload_dir_trailing_slash(
        self,
        runner: NmpCliRunner,
        files_client: FilesClient,
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
        file_paths = _list_paths(files_client, test_fileset["workspace"], test_fileset["name"])
        assert "file1.txt" in file_paths
        assert "file2.txt" in file_paths

    def test_upload_to_nonexistent_fileset_fails(
        self,
        runner: NmpCliRunner,
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
        runner: NmpCliRunner,
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
        runner: NmpCliRunner,
        files_client: FilesClient,
        random_workspace: str,
        tmp_path: Path,
    ):
        """Test that uploads without a fileset name create a new fileset."""
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
        fileset = files_client.get_fileset(name=fileset_name, workspace=random_workspace).data()
        assert fileset.name == fileset_name

        # Verify file was uploaded
        assert "test.txt" in _list_paths(files_client, random_workspace, fileset_name)

    def test_upload_without_workspace_is_usage_error(self, runner: NmpCliRunner, tmp_path: Path):
        test_file = tmp_path / "test.txt"
        test_file.write_text("Hello!")

        # The injected client has no default workspace and none was passed.
        result = runner.invoke(app, f"files upload {test_file} some-fileset")

        assert_exit_code(result, 2)
        assert "Missing workspace" in result.stderr


@pytest.fixture
def fileset_with_nested_files(files_client: FilesClient, random_workspace: str, tmp_path: Path) -> dict:
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

    for local in dir_a.rglob("*"):
        if local.is_file():
            files_client.upload_file(
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
        runner: NmpCliRunner,
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
        runner: NmpCliRunner,
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
        runner: NmpCliRunner,
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

    def test_download_missing_fileset_is_remote_error(
        self, runner: NmpCliRunner, random_workspace: str, tmp_path: Path
    ):
        result = runner.invoke(app, f"files download nope --workspace {random_workspace} -o {tmp_path}/")

        assert_exit_code(result, 3)
        assert "not found" in result.stderr.lower()


class TestFilesListDelete:
    """Tests for the files list and delete commands."""

    def test_list_files(self, runner: NmpCliRunner, fileset_with_nested_files: dict):
        result = runner.invoke(
            app,
            f"files list {fileset_with_nested_files['name']} --workspace {fileset_with_nested_files['workspace']}",
        )

        assert_exit_code(result, 0)
        listed = json.loads(result.stdout)
        assert sorted(item["path"] for item in listed) == ["a/b/file2.txt", "a/b/file3.txt", "a/file1.txt"]
        assert all(item["size"] == 8 for item in listed)

    def test_list_files_remote_path_prefix(self, runner: NmpCliRunner, fileset_with_nested_files: dict):
        result = runner.invoke(
            app,
            f"files list {fileset_with_nested_files['name']} --workspace {fileset_with_nested_files['workspace']} --remote-path a/b/",
        )

        assert_exit_code(result, 0)
        assert sorted(item["path"] for item in json.loads(result.stdout)) == ["a/b/file2.txt", "a/b/file3.txt"]

    def test_list_files_glob(self, runner: NmpCliRunner, fileset_with_nested_files: dict):
        result = runner.invoke(
            app,
            f"files list {fileset_with_nested_files['name']} --workspace {fileset_with_nested_files['workspace']} --remote-path a/b/*2.txt",
        )

        assert_exit_code(result, 0)
        assert [item["path"] for item in json.loads(result.stdout)] == ["a/b/file2.txt"]

    def test_list_files_table_columns(self, runner: NmpCliRunner, fileset_with_nested_files: dict):
        result = runner.invoke(
            app,
            f"files list {fileset_with_nested_files['name']} --workspace {fileset_with_nested_files['workspace']} -f table",
        )

        assert_exit_code(result, 0)
        assert "PATH" in result.stdout and "SIZE" in result.stdout
        assert "a/b/file2.txt" in result.stdout

    def test_list_files_missing_fileset(self, runner: NmpCliRunner, random_workspace: str):
        result = runner.invoke(app, f"files list nope --workspace {random_workspace}")

        assert_exit_code(result, 3)
        assert "not found" in result.stderr.lower()

    def test_delete_file(self, runner: NmpCliRunner, files_client: FilesClient, fileset_with_nested_files: dict):
        result = runner.invoke(
            app,
            f"files delete {fileset_with_nested_files['name']} --workspace {fileset_with_nested_files['workspace']} --remote-path a/b/file2.txt",
        )

        assert_exit_code(result, 0)
        assert f"Deleted {fileset_with_nested_files['name']}#a/b/file2.txt" in result.stdout
        remaining = _list_paths(files_client, fileset_with_nested_files["workspace"], fileset_with_nested_files["name"])
        assert sorted(remaining) == ["a/b/file3.txt", "a/file1.txt"]

    def test_delete_missing_file(self, runner: NmpCliRunner, fileset_with_nested_files: dict):
        result = runner.invoke(
            app,
            f"files delete {fileset_with_nested_files['name']} --workspace {fileset_with_nested_files['workspace']} --remote-path a/nope.txt",
        )

        assert_exit_code(result, 3)
        assert "not found" in result.stderr.lower()


def _fileset_name() -> str:
    return f"fs-{uuid.uuid4().hex[:8]}"


class TestFilesetsCrud:
    """Tests for the files filesets subcommands."""

    def test_filesets_lifecycle(self, runner: NmpCliRunner, random_workspace: str):
        name = _fileset_name()

        result = runner.invoke(
            app,
            [
                "files",
                "filesets",
                "create",
                name,
                "--description",
                "demo",
                "--purpose",
                "dataset",
                "--custom-fields",
                '{"team": "nlp"}',
                "--workspace",
                random_workspace,
            ],
        )
        assert_exit_code(result, 0)
        created = json.loads(result.stdout)
        assert created["name"] == name
        assert created["workspace"] == random_workspace
        assert created["description"] == "demo"
        assert created["purpose"] == "dataset"
        assert created["custom_fields"] == {"team": "nlp"}

        result = runner.invoke(app, ["files", "filesets", "get", name, "--workspace", random_workspace])
        assert_exit_code(result, 0)
        assert json.loads(result.stdout)["name"] == name

        result = runner.invoke(app, ["files", "filesets", "list", "--workspace", random_workspace])
        assert_exit_code(result, 0)
        listed = json.loads(result.stdout)
        assert [item["name"] for item in listed["data"]] == [name]
        assert listed["pagination"]["total_results"] == 1

        result = runner.invoke(
            app,
            [
                "files",
                "filesets",
                "update",
                name,
                "--description",
                "changed",
                "--purpose",
                "model",
                "--workspace",
                random_workspace,
            ],
        )
        assert_exit_code(result, 0)
        updated = json.loads(result.stdout)
        assert updated["description"] == "changed"
        assert updated["purpose"] == "model"
        assert updated["custom_fields"] == {"team": "nlp"}

        result = runner.invoke(app, ["files", "filesets", "delete", name, "--workspace", random_workspace])
        assert_exit_code(result, 0)
        assert "Deleted successfully" in result.stdout

        result = runner.invoke(app, ["files", "filesets", "get", name, "--workspace", random_workspace])
        assert_exit_code(result, 3)
        assert "Not found" in result.stderr

    def test_filesets_create_from_stdin(self, runner: NmpCliRunner, random_workspace: str):
        name = _fileset_name()
        result = runner.invoke(
            app,
            ["files", "filesets", "create", name, "--input-file", "-", "--workspace", random_workspace],
            input='{"description": "piped"}\n',
        )
        assert_exit_code(result, 0)
        assert json.loads(result.stdout)["description"] == "piped"

    def test_filesets_create_exist_ok_returns_existing(self, runner: NmpCliRunner, random_workspace: str):
        name = _fileset_name()
        first = runner.invoke(
            app, ["files", "filesets", "create", name, "--description", "first", "--workspace", random_workspace]
        )
        assert_exit_code(first, 0)

        conflict = runner.invoke(app, ["files", "filesets", "create", name, "--workspace", random_workspace])
        assert_exit_code(conflict, 3)
        assert "Conflict" in conflict.stderr

        result = runner.invoke(
            app, ["files", "filesets", "create", name, "--exist-ok", "--workspace", random_workspace]
        )
        assert_exit_code(result, 0)
        existing = json.loads(result.stdout)
        assert existing["id"] == json.loads(first.stdout)["id"]
        assert existing["description"] == "first"

    def test_filesets_create_rejects_invalid_name_locally(self, runner: NmpCliRunner, random_workspace: str):
        result = runner.invoke(app, ["files", "filesets", "create", "X", "--workspace", random_workspace])
        assert_exit_code(result, 2)
        assert "Invalid input" in result.stderr
        assert "name" in result.stderr

    def test_filesets_list_filters_and_pages(self, runner: NmpCliRunner, random_workspace: str):
        names = sorted(_fileset_name() for _ in range(3))
        for index, name in enumerate(names):
            purpose = "dataset" if index == 0 else "generic"
            assert_exit_code(
                runner.invoke(
                    app,
                    ["files", "filesets", "create", name, "--purpose", purpose, "--workspace", random_workspace],
                ),
                0,
            )

        result = runner.invoke(
            app, ["files", "filesets", "list", "--workspace", random_workspace, "--page-size", "2", "--sort", "name"]
        )
        assert_exit_code(result, 0)
        first_page = json.loads(result.stdout)
        assert [item["name"] for item in first_page["data"]] == names[:2]
        assert first_page["pagination"]["total_pages"] == 2
        assert "More pages" in result.stderr

        result = runner.invoke(
            app,
            ["files", "filesets", "list", "--workspace", random_workspace, "--page-size", "2", "--all-pages"],
        )
        assert_exit_code(result, 0)
        all_pages = json.loads(result.stdout)
        assert sorted(item["name"] for item in all_pages["data"]) == names
        assert all_pages["pagination"]["total_results"] == 3
        assert "More pages" not in result.stderr

        result = runner.invoke(
            app, ["files", "filesets", "list", "--workspace", random_workspace, "--filter.purpose", "dataset"]
        )
        assert_exit_code(result, 0)
        assert [item["name"] for item in json.loads(result.stdout)["data"]] == [names[0]]

        result = runner.invoke(
            app, ["files", "filesets", "list", "--workspace", random_workspace, "--filter.name", names[1]]
        )
        assert_exit_code(result, 0)
        assert [item["name"] for item in json.loads(result.stdout)["data"]] == [names[1]]

    def test_filesets_code_output_sends_nothing(self, runner: NmpCliRunner, random_workspace: str):
        result = runner.invoke(
            app, ["files", "filesets", "create", "code-fileset", "--workspace", random_workspace, "-f", "code"]
        )
        assert_exit_code(result, 0)
        assert "from nemo_platform_plugin.files.client import FilesClient" in result.stdout
        assert "client.create_fileset(" in result.stdout
        assert "NeMoPlatform" not in result.stdout

        result = runner.invoke(app, ["files", "filesets", "list", "--workspace", random_workspace])
        assert json.loads(result.stdout)["data"] == []
