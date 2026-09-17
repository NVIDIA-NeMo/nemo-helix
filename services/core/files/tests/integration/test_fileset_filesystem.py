# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Integration tests for FilesetFileSystem fsspec implementation.

These tests verify the fsspec filesystem interface works correctly
with the Files service, testing:
- File listing (ls)
- File reading (open, cat)
- File writing (pipe, open with write mode)
- File deletion (rm)
- Directory emulation
- Range requests for parquet files
"""

from pathlib import Path
from typing import Iterator
from unittest.mock import patch

import anyio
import duckdb
import fsspec
import pandas as pd
import pytest
from filesets import (
    AsyncFilesetFileSystem,
    FilesetFileSystem,
    FilesetPathError,
    build_fileset_ref,
    parse_fileset_path,
    parse_fileset_ref,
)
from nemo_platform import NeMoPlatform
from nemo_platform_plugin.client.adapter import client_from_platform
from nemo_platform_plugin.files.client import AsyncFilesClient, FilesClient
from nemo_platform_plugin.files.types import FilesetOutput


class TestParseFilesetRef:
    """Unit tests for parse_fileset_ref function."""

    @pytest.mark.parametrize(
        "ref,expected",
        [
            # New format with # separator (workspace in path)
            (
                "default/my-fileset#data/file.txt",
                ("default", "my-fileset", "data/file.txt"),
            ),
            ("default/my-fileset#file.txt", ("default", "my-fileset", "file.txt")),
            ("default/my-fileset#", ("default", "my-fileset", "")),
            ("default/my-fileset", ("default", "my-fileset", "")),
            # URL format with fileset:// prefix
            (
                "fileset://default/my-fileset#file.txt",
                ("default", "my-fileset", "file.txt"),
            ),
            ("fileset://default/my-fileset", ("default", "my-fileset", "")),
            ("fileset://default/my-fileset#", ("default", "my-fileset", "")),
            # Legacy format (workspace/fileset/path with 3+ segments, no #)
            (
                "default/my-fileset/data/file.txt",
                ("default", "my-fileset", "data/file.txt"),
            ),
            ("default/my-fileset/file.txt", ("default", "my-fileset", "file.txt")),
            # Leading slashes stripped
            ("/default/my-fileset#file.txt", ("default", "my-fileset", "file.txt")),
            ("//default/my-fileset#file.txt", ("default", "my-fileset", "file.txt")),
        ],
    )
    def test_parse_fileset_ref(self, ref: str, expected: tuple[str, str, str]):
        """Test parsing various fileset reference formats."""
        # These refs all have workspace in them, so workspace_fallback=None is correct
        assert parse_fileset_ref(ref, workspace_fallback=None) == expected

    @pytest.mark.parametrize(
        "ref,error_match",
        [
            # Empty refs
            ("", "must include fileset name"),
            ("fileset://", "must include fileset name"),
            # Missing workspace (no fallback provided)
            ("my-fileset", "Workspace required"),
            ("my-fileset#file.txt", "Workspace required"),
            ("fileset://my-fileset", "Workspace required"),
        ],
    )
    def test_parse_fileset_ref_errors(self, ref: str, error_match: str):
        """Test that invalid refs raise FilesetPathError."""
        with pytest.raises(FilesetPathError, match=error_match):
            parse_fileset_ref(ref, workspace_fallback=None)

    @pytest.mark.parametrize(
        "ref,fallback,expected",
        [
            # Fallback used when workspace not in path (hash format only)
            ("my-fileset#file.txt", "default", ("default", "my-fileset", "file.txt")),
            ("my-fileset", "default", ("default", "my-fileset", "")),
            # Note: URL format (fileset://) requires explicit workspace; fallback not used
            # Fallback ignored when workspace is in path
            ("ws/fs#file.txt", "other", ("ws", "fs", "file.txt")),
            ("ws/fs", "other", ("ws", "fs", "")),
        ],
    )
    def test_parse_fileset_ref_workspace_fallback(self, ref: str, fallback: str | None, expected: tuple[str, str, str]):
        """Test workspace_fallback parameter."""
        assert parse_fileset_ref(ref, workspace_fallback=fallback) == expected


class TestParseFilesetPath:
    """Unit tests for parse_fileset_path function.

    parse_fileset_path is designed for SDK methods where the path parameter
    could be either a fileset ref or a pure file path. It uses a simple rule:
    - If path contains '#' or starts with 'fileset://', it's a fileset ref
    - Otherwise, it's a pure file path (requires workspace_fallback)
    """

    @pytest.mark.parametrize(
        "path,expected",
        [
            # Fileset refs (with # or fileset://) - delegates to parse_fileset_ref
            (
                "workspace/fileset#data/file.txt",
                ("workspace", "fileset", "data/file.txt"),
            ),
            ("workspace/fileset#", ("workspace", "fileset", "")),
            (
                "fileset://workspace/fileset#file.txt",
                ("workspace", "fileset", "file.txt"),
            ),
            ("fileset://workspace/fileset", ("workspace", "fileset", "")),
        ],
    )
    def test_parse_fileset_path(self, path: str, expected: tuple[str, str, str]):
        """Test that parse_fileset_path correctly parses fileset refs."""
        assert parse_fileset_path(path, workspace_fallback=None) == expected

    def test_parse_fileset_path_requires_fallback_for_pure_paths(self):
        """Test that parse_fileset_path raises error for pure paths without fallback."""
        with pytest.raises(FilesetPathError):
            parse_fileset_path("data/file.txt", workspace_fallback=None)

    @pytest.mark.parametrize(
        "path,fallback,expected",
        [
            # With fileset ref, fallback is used for workspace if missing
            ("my-fileset#file.txt", "default", ("default", "my-fileset", "file.txt")),
            # With pure path, fallback is used for workspace, fileset is None
            ("data/file.txt", "default", ("default", None, "data/file.txt")),
            ("file.txt", "default", ("default", None, "file.txt")),
            ("", "default", ("default", None, "")),
        ],
    )
    def test_parse_fileset_path_with_fallback(self, path: str, fallback: str, expected: tuple[str, str | None, str]):
        """Test workspace_fallback is applied for both refs and pure paths."""
        assert parse_fileset_path(path, workspace_fallback=fallback) == expected


class TestBuildFilesetRef:
    """Unit tests for build_fileset_ref function."""

    @pytest.mark.parametrize(
        "ref,workspace,expected",
        [
            # With workspace in ref
            (
                "default/my-fileset#data/file.txt",
                None,
                "default/my-fileset#data/file.txt",
            ),
            ("default/my-fileset#file.txt", None, "default/my-fileset#file.txt"),
            ("default/my-fileset", None, "default/my-fileset"),
            # Legacy format normalized
            (
                "default/my-fileset/data/file.txt",
                None,
                "default/my-fileset#data/file.txt",
            ),
            # Without workspace, using fallback
            ("my-fileset#file.txt", "default", "default/my-fileset#file.txt"),
            ("my-fileset", "default", "default/my-fileset"),
            # Fallback workspace ignored when ref has workspace
            ("ws/fs#file.txt", "other", "ws/fs#file.txt"),
        ],
    )
    def test_build_fileset_ref(self, ref: str, workspace: str | None, expected: str):
        """Test building refs to canonical format."""
        assert build_fileset_ref(ref, workspace=workspace) == expected

    def test_build_fileset_ref_without_workspace_raises(self):
        """Test that refs without workspace raise FilesetPathError when no fallback."""
        with pytest.raises(FilesetPathError, match="Workspace required"):
            build_fileset_ref("my-fileset#file.txt")

    @pytest.mark.parametrize(
        "ref,workspace,fileset,expected",
        [
            # Construct ref from file path
            ("data/file.txt", "ws", "fs", "ws/fs#data/file.txt"),
            ("file.txt", "ws", "fs", "ws/fs#file.txt"),
            # Empty file path = fileset root
            ("", "ws", "fs", "ws/fs"),
            # Leading slashes stripped
            ("/data/file.txt", "ws", "fs", "ws/fs#data/file.txt"),
            # Nested paths
            ("a/b/c/d.txt", "ws", "fs", "ws/fs#a/b/c/d.txt"),
        ],
    )
    def test_build_fileset_ref_with_fileset(self, ref: str, workspace: str, fileset: str, expected: str):
        """Test constructing refs from file paths using fileset parameter."""
        result = build_fileset_ref(ref, workspace=workspace, fileset=fileset)
        assert result == expected

    def test_build_fileset_ref_fileset_requires_workspace(self):
        """Test that fileset without workspace raises."""
        with pytest.raises(FilesetPathError, match="workspace required"):
            build_fileset_ref("data/file.txt", fileset="fs")

    def test_build_fileset_ref_with_hash_ignores_fileset(self):
        """Test that refs with # are parsed normally even with fileset."""
        # When ref contains #, it's parsed as a fileset ref, not a file path
        result = build_fileset_ref(
            "ws/actual-fs#data/file.txt",
            workspace="other-ws",
            fileset="ignored-fs",
        )
        assert result == "ws/actual-fs#data/file.txt"


class TestFilesetParent:
    """Unit tests for FilesetFileSystem._parent method."""

    @pytest.mark.parametrize(
        "path,expected",
        [
            # File in subdirectory -> parent is subdirectory
            ("default/my-fileset#data/file.txt", "default/my-fileset#data"),
            ("default/my-fileset#a/b/c/file.txt", "default/my-fileset#a/b/c"),
            # File at fileset root -> parent is fileset root
            ("default/my-fileset#file.txt", "default/my-fileset"),
            ("default/my-fileset#file", "default/my-fileset"),
            # Subdirectory -> parent is parent directory or fileset root
            ("default/my-fileset#data/subdir", "default/my-fileset#data"),
            ("default/my-fileset#data", "default/my-fileset"),
            # Fileset root -> parent is itself (like / in Unix)
            ("default/my-fileset", "default/my-fileset"),
            ("default/my-fileset#", "default/my-fileset"),
        ],
    )
    def test_fileset_parent(self, path: str, expected: str):
        """Test _parent returns correct parent path."""
        assert FilesetFileSystem._parent(path) == expected

    def test_fileset_parent_without_workspace_raises(self):
        """Test that paths without workspace raise FilesetPathError."""
        with pytest.raises(FilesetPathError, match="Workspace required"):
            FilesetFileSystem._parent("my-fileset#file.txt")

    def test_fileset_parent_without_workspace_raises_even_for_root(self):
        """Test that fileset root without workspace raises FilesetPathError."""
        with pytest.raises(FilesetPathError, match="Workspace required"):
            FilesetFileSystem._parent("my-fileset")


@pytest.fixture(autouse=True, scope="module")
def register_fileset_protocol():
    """Register the fileset:// protocol with fsspec before any tests run."""
    FilesetFileSystem.register_fsspec()


@pytest.fixture
def sample_dataset(tmp_path: Path) -> Iterator[Path]:
    """Create a sample dataset directory structure for testing uploads/downloads.

    Structure:
        my_dataset/
        ├── README.md
        ├── data/
        │   ├── train.csv
        │   └── test.csv
        └── config/
            └── settings.json
    """
    dataset_path = tmp_path / "my_dataset"
    dataset_path.mkdir()
    (dataset_path / "README.md").write_bytes(b"# My Dataset")
    (dataset_path / "data").mkdir()
    (dataset_path / "data" / "train.csv").write_bytes(b"id,value\n1,100\n2,200")
    (dataset_path / "data" / "test.csv").write_bytes(b"id,value\n3,300")
    (dataset_path / "config").mkdir()
    (dataset_path / "config" / "settings.json").write_bytes(b'{"batch_size": 32}')
    yield dataset_path


class TestFilesetFileSystem:
    """Test fsspec operations via FilesetFileSystem."""

    @pytest.fixture
    def fs(self, sdk: NeMoPlatform) -> FilesetFileSystem:
        """Create a FilesetFileSystem backed by the test SDK."""
        return FilesetFileSystem(client=client_from_platform(sdk, FilesClient))

    def test_ls_empty_fileset(self, fs: FilesetFileSystem, fileset: FilesetOutput):
        """Test listing an empty fileset."""
        path = f"{fileset.workspace}/{fileset.name}"
        result = fs.ls(path)
        assert result == []

    def test_ls_with_files(self, fs: FilesetFileSystem, fileset: FilesetOutput):
        """Test listing a fileset with files."""
        base = f"{fileset.workspace}/{fileset.name}"
        fs.pipe(f"{base}/file1.txt", b"content1")
        fs.pipe(f"{base}/file2.txt", b"content2")

        result = fs.ls(base, detail=True)

        assert len(result) == 2
        names = {r["name"] for r in result}
        assert f"{base}#file1.txt" in names
        assert f"{base}#file2.txt" in names

    def test_ls_with_directories(self, fs: FilesetFileSystem, fileset: FilesetOutput):
        """Test that nested files show as directories in listing."""
        base = f"{fileset.workspace}/{fileset.name}"
        fs.pipe(f"{base}/root.txt", b"root")
        fs.pipe(f"{base}/subdir/nested.txt", b"nested")

        result = fs.ls(base, detail=True)

        # Should see root.txt as file and subdir as directory
        assert len(result) == 2
        types = {r["name"]: r["type"] for r in result}
        assert types[f"{base}#root.txt"] == "file"
        assert types[f"{base}#subdir"] == "directory"

    def test_ls_subdirectory(self, fs: FilesetFileSystem, fileset: FilesetOutput):
        """Test listing files in a subdirectory."""
        base = f"{fileset.workspace}/{fileset.name}"
        fs.pipe(f"{base}/subdir/file1.txt", b"content1")
        fs.pipe(f"{base}/subdir/file2.txt", b"content2")

        result = fs.ls(f"{base}/subdir", detail=False)

        assert len(result) == 2
        assert f"{base}#subdir/file1.txt" in result
        assert f"{base}#subdir/file2.txt" in result

    def test_ls_subdirectory_trailing_slash(self, fs: FilesetFileSystem, fileset: FilesetOutput):
        """Test that ls with trailing slash returns same result as without.

        ls("workspace/fileset/subdir/") should return
        the same result as ls("workspace/fileset/subdir").
        """
        base = f"{fileset.workspace}/{fileset.name}"
        fs.pipe(f"{base}/subdir/file1.txt", b"content1")
        fs.pipe(f"{base}/subdir/file2.txt", b"content2")

        # ls without trailing slash
        result_no_slash = fs.ls(f"{base}/subdir", detail=False)
        # ls with trailing slash
        result_with_slash = fs.ls(f"{base}/subdir/", detail=False)

        assert len(result_no_slash) == 2
        assert len(result_with_slash) == 2
        assert set(result_no_slash) == set(result_with_slash)

    def test_cat_file(self, fs: FilesetFileSystem, fileset: FilesetOutput):
        """Test reading file content with cat."""
        content = b"Hello, fsspec!"
        path = f"{fileset.workspace}/{fileset.name}/test.txt"
        fs.pipe(path, content)

        result = fs.cat(path)

        assert result == content

    def test_cat_file_with_range(self, fs: FilesetFileSystem, fileset: FilesetOutput):
        """Test reading partial file content with byte range."""
        content = b"0123456789ABCDEF"
        path = f"{fileset.workspace}/{fileset.name}/test.txt"
        fs.pipe(path, content)

        result = fs.cat_file(path, start=4, end=10)

        assert result == b"456789"

    def test_open_read(self, fs: FilesetFileSystem, fileset: FilesetOutput):
        """Test opening a file for reading."""
        content = b"File content for reading"
        path = f"{fileset.workspace}/{fileset.name}/readable.txt"
        fs.pipe(path, content)

        with fs.open(path, "rb") as f:
            result = f.read()

        assert result == content

    def test_open_write(self, fs: FilesetFileSystem, fileset: FilesetOutput):
        """Test opening a file for writing."""
        path = f"{fileset.workspace}/{fileset.name}/writable.txt"
        content = b"Written via fsspec"

        with fs.open(path, "wb") as f:
            f.write(content)

        # Verify by reading back
        result = fs.cat(path)
        assert result == content

    def test_pipe(self, fs: FilesetFileSystem, fileset: FilesetOutput):
        """Test writing file content with pipe."""
        path = f"{fileset.workspace}/{fileset.name}/piped.txt"
        content = b"Piped content"

        fs.pipe(path, content)

        result = fs.cat(path)
        assert result == content

    def test_put_file(self, fs: FilesetFileSystem, fileset: FilesetOutput, tmp_path: Path):
        """Test uploading a local file with put_file."""
        # Create a local file
        local_file = tmp_path / "upload.txt"
        content = b"Content to upload via put_file"
        local_file.write_bytes(content)

        # Upload to fileset
        remote_path = f"{fileset.workspace}/{fileset.name}/uploaded.txt"
        fs.put_file(str(local_file), remote_path)

        # Verify content was uploaded
        result = fs.cat(remote_path)
        assert result == content

    def test_put_file_nested_path(self, fs: FilesetFileSystem, fileset: FilesetOutput, tmp_path: Path):
        """Test uploading a file to a nested path with put_file."""
        # Create a local file
        local_file = tmp_path / "nested_upload.txt"
        content = b"Nested upload content"
        local_file.write_bytes(content)

        # Upload to nested path in fileset
        remote_path = f"{fileset.workspace}/{fileset.name}/subdir/nested/uploaded.txt"
        fs.put_file(str(local_file), remote_path)

        # Verify content was uploaded
        result = fs.cat(remote_path)
        assert result == content

        # Verify parent directories are emulated
        parent_info = fs.info(f"{fileset.workspace}/{fileset.name}/subdir/nested")
        assert parent_info["type"] == "directory"

    def test_rm(self, fs: FilesetFileSystem, fileset: FilesetOutput):
        """Test deleting a file."""
        path = f"{fileset.workspace}/{fileset.name}/to_delete.txt"
        fs.pipe(path, b"delete me")

        assert fs.exists(path)

        fs.rm(path)

        assert not fs.exists(path)

    def test_info(self, fs: FilesetFileSystem, fileset: FilesetOutput):
        """Test getting file info."""
        content = b"Content for info test"
        path = f"{fileset.workspace}/{fileset.name}/info.txt"
        fs.pipe(path, content)

        info = fs.info(path)

        # info["name"] uses canonical format with # separator
        expected_name = f"{fileset.workspace}/{fileset.name}#info.txt"
        assert info["name"] == expected_name
        assert info["size"] == len(content)
        assert info["type"] == "file"

    def test_info_directory(self, fs: FilesetFileSystem, fileset: FilesetOutput):
        """Test getting info for fileset root (directory)."""
        path = f"{fileset.workspace}/{fileset.name}"
        info = fs.info(path)

        assert info["type"] == "directory"

    def test_exists(self, fs: FilesetFileSystem, fileset: FilesetOutput):
        """Test checking file existence."""
        base = f"{fileset.workspace}/{fileset.name}"
        fs.pipe(f"{base}/exists.txt", b"I exist")

        assert fs.exists(f"{base}/exists.txt")
        assert not fs.exists(f"{base}/does_not_exist.txt")

    def test_parquet_read(self, fs: FilesetFileSystem, fileset: FilesetOutput):
        """Test reading a parquet file via fsspec.

        This validates that range requests work correctly for formats
        like Parquet that read file footers.
        """
        df = pd.DataFrame(
            {
                "id": [1, 2, 3],
                "name": ["Alice", "Bob", "Charlie"],
                "value": [100.5, 200.0, 300.25],
            }
        )
        parquet_bytes = df.to_parquet(index=False)

        path = f"{fileset.workspace}/{fileset.name}/data.parquet"
        fs.pipe(path, parquet_bytes)

        # Read parquet via fsspec
        result = pd.read_parquet(fs.open(path, "rb"))

        pd.testing.assert_frame_equal(result, df)

    def test_protocol_url(self, fs: FilesetFileSystem, fileset: FilesetOutput):
        """Test that protocol prefix is handled correctly."""
        content = b"Protocol test"
        base = f"{fileset.workspace}/{fileset.name}"
        fs.pipe(f"{base}/protocol.txt", content)

        # Both with and without protocol should work
        path_no_proto = f"{base}/protocol.txt"
        path_with_proto = f"fileset://{base}/protocol.txt"

        assert fs.cat(path_no_proto) == content
        assert fs.cat(path_with_proto) == content

    def test_fsspec_filesystem_registration(
        self,
        sdk: NeMoPlatform,
        fileset: FilesetOutput,
    ):
        """Test that FilesetFileSystem can be instantiated via fsspec.filesystem()."""
        # Protocol is registered by the autouse fixture
        fs = fsspec.filesystem(
            "fileset",
            client=client_from_platform(sdk, FilesClient),
            skip_instance_cache=True,
        )

        assert isinstance(fs, FilesetFileSystem)

        # Verify it works by listing an empty fileset
        path = f"{fileset.workspace}/{fileset.name}"
        result = fs.ls(path)
        assert result == []

    def test_find(self, fs: FilesetFileSystem, fileset: FilesetOutput):
        """Test recursive file discovery with find."""
        base = f"{fileset.workspace}/{fileset.name}"
        fs.pipe(f"{base}/root.txt", b"root")
        fs.pipe(f"{base}/dir1/file1.txt", b"file1")
        fs.pipe(f"{base}/dir1/file2.txt", b"file2")
        fs.pipe(f"{base}/dir1/subdir/nested.txt", b"nested")
        fs.pipe(f"{base}/dir2/other.txt", b"other")

        # Find all files recursively
        result = fs.find(base)

        assert len(result) == 5
        assert f"{base}#root.txt" in result
        assert f"{base}#dir1/file1.txt" in result
        assert f"{base}#dir1/subdir/nested.txt" in result

    def test_glob(self, fs: FilesetFileSystem, fileset: FilesetOutput):
        """Test pattern matching with glob."""
        base = f"{fileset.workspace}/{fileset.name}"
        fs.pipe(f"{base}/data.csv", b"csv")
        fs.pipe(f"{base}/data.json", b"json")
        fs.pipe(f"{base}/config.json", b"config")
        fs.pipe(f"{base}/subdir/nested.json", b"nested")

        # Use find() to get all files, then filter by extension
        # (glob pattern matching with custom path format is complex due to fnmatch requirements)
        all_files = fs.find(base)
        json_at_root = [f for f in all_files if f.endswith(".json") and "/" not in f.split("#")[-1]]
        json_all = [f for f in all_files if f.endswith(".json")]

        assert len(json_at_root) == 2
        assert f"{base}#data.json" in json_at_root
        assert f"{base}#config.json" in json_at_root

        # All json files including nested
        assert len(json_all) == 3

    def test_isdir_isfile(self, fs: FilesetFileSystem, fileset: FilesetOutput):
        """Test isdir and isfile type checking."""
        base = f"{fileset.workspace}/{fileset.name}"
        fs.pipe(f"{base}/file.txt", b"content")
        fs.pipe(f"{base}/subdir/nested.txt", b"nested")

        # File checks
        assert fs.isfile(f"{base}/file.txt")
        assert not fs.isdir(f"{base}/file.txt")

        # Directory checks (fileset root)
        assert fs.isdir(base)
        assert not fs.isfile(base)

        # Subdirectory checks
        assert fs.isdir(f"{base}/subdir")
        assert not fs.isfile(f"{base}/subdir")

    def test_cat_multiple_files(self, fs: FilesetFileSystem, fileset: FilesetOutput):
        """Test reading multiple files at once with cat."""
        base = f"{fileset.workspace}/{fileset.name}"
        fs.pipe(f"{base}/file1.txt", b"content1")
        fs.pipe(f"{base}/file2.txt", b"content2")
        fs.pipe(f"{base}/file3.txt", b"content3")

        # Read multiple files using new path format
        paths = [f"{base}#file1.txt", f"{base}#file2.txt", f"{base}#file3.txt"]
        result = fs.cat(paths)

        assert isinstance(result, dict)
        assert result[f"{base}#file1.txt"] == b"content1"
        assert result[f"{base}#file2.txt"] == b"content2"
        assert result[f"{base}#file3.txt"] == b"content3"

    def test_walk(self, fs: FilesetFileSystem, fileset: FilesetOutput):
        """Test directory traversal with walk."""
        base = f"{fileset.workspace}/{fileset.name}"
        fs.pipe(f"{base}/root.txt", b"root")
        fs.pipe(f"{base}/dir1/file1.txt", b"file1")
        fs.pipe(f"{base}/dir1/subdir/nested.txt", b"nested")

        # Walk the directory tree
        walked = list(fs.walk(base))

        # Should have entries for root, dir1, and dir1/subdir
        assert len(walked) >= 1
        # First entry should be the root - files includes fileset#filename format
        _, _, files = walked[0]
        # Walk returns files with fileset#path format relative to walked directory
        assert any("root.txt" in f for f in files)

    def test_head_tail(self, fs: FilesetFileSystem, fileset: FilesetOutput):
        """Test reading first/last bytes of a file."""
        content = b"0123456789ABCDEFGHIJ"
        path = f"{fileset.workspace}/{fileset.name}/test.txt"
        fs.pipe(path, content)

        # Read first 5 bytes
        result = fs.head(path, size=5)
        assert result == b"01234"

        # Read last 5 bytes
        result = fs.tail(path, size=5)
        assert result == b"FGHIJ"

    def test_get_single_file(self, fs: FilesetFileSystem, fileset: FilesetOutput, tmp_path: Path):
        """Test downloading a single file with get().

        Tests three behaviors:
        - lpath="foo" (no slash, doesn't exist): creates a file named "foo"
        - lpath="foo/" (with slash): creates directory "foo" and puts file inside
        - lpath="existing_dir" (no slash, exists): puts file inside existing directory
        """
        content = b"Single file content"
        base = f"{fileset.workspace}/{fileset.name}"
        fs.pipe(f"{base}/data/test.txt", content)

        # Test 1: lpath without trailing slash - creates file named "renamed"
        renamed_file = tmp_path / "renamed"
        fs.get(f"{base}/data/test.txt", str(renamed_file))
        assert renamed_file.is_file()
        assert renamed_file.read_bytes() == content

        # Test 2: lpath with trailing slash - creates directory and preserves filename
        target_dir = tmp_path / "target_dir"
        target_dir.mkdir()
        fs.get(f"{base}/data/test.txt", str(target_dir) + "/")
        assert (target_dir / "test.txt").is_file()
        assert (target_dir / "test.txt").read_bytes() == content

        # Test 3: lpath is existing directory (no slash) - puts file inside
        existing_dir = tmp_path / "existing_dir"
        existing_dir.mkdir()
        fs.get(f"{base}/data/test.txt", str(existing_dir))
        assert (existing_dir / "test.txt").is_file()
        assert (existing_dir / "test.txt").read_bytes() == content

    def test_get_trailing_slash_semantics(self, fs: FilesetFileSystem, fileset: FilesetOutput, tmp_path: Path):
        """Test trailing slash semantics for get() per fsspec docs.

        From https://filesystem-spec.readthedocs.io/en/latest/copying.html:

        Source trailing slash controls whether to preserve source directory name:
        - "source/subdir/" (with slash) -> copy CONTENTS directly into dest
        - "source/subdir" (no slash) -> copy directory ITSELF (preserve name)

        Dest trailing slash indicates dest is a directory (for file placement),
        but does NOT affect whether source directory name is preserved.

        Examples from docs:
        1a. cp("source/file", "target/") -> target/file
        1c. cp("source/file", "target/newfile") -> target/newfile
        1e. cp("source/subdir/", "target/") -> target/ contains subdir contents
        1e variant. cp("source/subdir", "target/") -> target/subdir/ created
        """
        base = f"{fileset.workspace}/{fileset.name}"

        # Create a directory structure
        fs.pipe(f"{base}/subdir/file1.txt", b"file1")
        fs.pipe(f"{base}/subdir/file2.txt", b"file2")
        fs.pipe(f"{base}/subdir/nested/deep.txt", b"deep")

        # Example 1e: Source WITH trailing slash -> copy CONTENTS directly
        # cp("source/subdir/", "target/") -> files go directly into target/
        dest_contents = tmp_path / "dest_contents"
        dest_contents.mkdir()
        fs.get(f"{base}/subdir/", str(dest_contents) + "/", recursive=True)

        assert (dest_contents / "file1.txt").read_bytes() == b"file1"
        assert (dest_contents / "file2.txt").read_bytes() == b"file2"
        assert (dest_contents / "nested" / "deep.txt").read_bytes() == b"deep"
        # Verify subdir name is NOT preserved (contents copied directly)
        assert not (dest_contents / "subdir").exists()

        # Example 1e variant: Source WITHOUT trailing slash -> preserve dir name
        # cp("source/subdir", "target/") -> target/subdir/ is created
        dest_preserve = tmp_path / "dest_preserve"
        dest_preserve.mkdir()
        fs.get(f"{base}/subdir", str(dest_preserve) + "/", recursive=True)

        # subdir name IS preserved
        assert (dest_preserve / "subdir").is_dir()
        assert (dest_preserve / "subdir" / "file1.txt").read_bytes() == b"file1"
        assert (dest_preserve / "subdir" / "file2.txt").read_bytes() == b"file2"
        assert (dest_preserve / "subdir" / "nested" / "deep.txt").read_bytes() == b"deep"

        # Dest without trailing slash, source without trailing slash
        # cp("source/subdir", "target") -> target/subdir/ is created
        dest_no_slash = tmp_path / "dest_no_slash"
        dest_no_slash.mkdir()
        fs.get(f"{base}/subdir", str(dest_no_slash), recursive=True)

        # subdir name IS preserved (same as above)
        assert (dest_no_slash / "subdir").is_dir()
        assert (dest_no_slash / "subdir" / "file1.txt").read_bytes() == b"file1"

        # Source with trailing slash, dest without trailing slash
        # cp("source/subdir/", "target") -> contents go into target/
        dest_source_slash = tmp_path / "dest_source_slash"
        dest_source_slash.mkdir()
        fs.get(f"{base}/subdir/", str(dest_source_slash), recursive=True)

        # Contents go directly into dest (no subdir nesting)
        assert (dest_source_slash / "file1.txt").read_bytes() == b"file1"
        assert not (dest_source_slash / "subdir").exists()

    def test_get_fileset_root(self, fs: FilesetFileSystem, fileset: FilesetOutput, tmp_path: Path):
        """Test downloading from fileset root path (workspace/fileset without subpath).

        When the source path is just the fileset root (no subpath), the trailing
        slash semantics still apply:
        - "workspace/fileset/" -> copy fileset contents directly into dest
        - "workspace/fileset" -> preserve fileset name as subdirectory in dest
        """
        base = f"{fileset.workspace}/{fileset.name}"

        # Create files at fileset root and in subdirectories
        fs.pipe(f"{base}/root_file.txt", b"root")
        fs.pipe(f"{base}/data/file1.txt", b"file1")
        fs.pipe(f"{base}/data/nested/deep.txt", b"deep")

        # Test 1: Fileset root WITH trailing slash -> copy contents directly
        dest_contents = tmp_path / "contents"
        dest_contents.mkdir()
        fs.get(f"{base}/", str(dest_contents) + "/", recursive=True)

        assert (dest_contents / "root_file.txt").read_bytes() == b"root"
        assert (dest_contents / "data" / "file1.txt").read_bytes() == b"file1"
        assert (dest_contents / "data" / "nested" / "deep.txt").read_bytes() == b"deep"
        # Fileset name should NOT be preserved
        assert not (dest_contents / fileset.name).exists()

        # Test 2: Fileset root WITHOUT trailing slash -> ALSO copies contents directly
        # (Special case: fileset root always copies contents directly, unlike subdirs)
        dest_no_slash = tmp_path / "no_slash"
        dest_no_slash.mkdir()
        fs.get(base, str(dest_no_slash) + "/", recursive=True)

        # Contents go directly into dest (no fileset name subfolder)
        assert (dest_no_slash / "root_file.txt").read_bytes() == b"root"
        assert (dest_no_slash / "data" / "file1.txt").read_bytes() == b"file1"
        assert (dest_no_slash / "data" / "nested" / "deep.txt").read_bytes() == b"deep"
        assert not (dest_no_slash / fileset.name).exists()

    def test_put_fileset_root(self, fs: FilesetFileSystem, fileset: FilesetOutput, tmp_path: Path):
        """Test uploading to fileset root path (workspace/fileset without subpath).

        When the dest path is just the fileset root (no subpath), the trailing
        slash semantics still apply:
        - "local_dir/" -> copy contents directly to fileset root
        - "local_dir" -> preserve local dir name as subdirectory in fileset
        """
        base = f"{fileset.workspace}/{fileset.name}"

        # Create local directory structure
        local_dir = tmp_path / "my_data"
        local_dir.mkdir()
        (local_dir / "file1.txt").write_bytes(b"file1")
        (local_dir / "file2.txt").write_bytes(b"file2")
        (local_dir / "subdir").mkdir()
        (local_dir / "subdir" / "nested.txt").write_bytes(b"nested")

        # Test 1: Source WITH trailing slash -> upload contents directly to fileset root
        fs.put(str(local_dir) + "/", f"{base}/", recursive=True)

        # Files should be at fileset root
        assert fs.cat(f"{base}/file1.txt") == b"file1"
        assert fs.cat(f"{base}/file2.txt") == b"file2"
        assert fs.cat(f"{base}/subdir/nested.txt") == b"nested"

        # Clean up for next test
        fs.rm(f"{base}/file1.txt")
        fs.rm(f"{base}/file2.txt")
        fs.rm(f"{base}/subdir/nested.txt")

        # Test 2: Source WITHOUT trailing slash -> preserve local dir name in fileset
        fs.put(str(local_dir), f"{base}/", recursive=True)

        # my_data/ should be preserved as subdirectory
        assert fs.cat(f"{base}/my_data/file1.txt") == b"file1"
        assert fs.cat(f"{base}/my_data/file2.txt") == b"file2"
        assert fs.cat(f"{base}/my_data/subdir/nested.txt") == b"nested"

    def test_put_single_file_to_fileset_root(self, fs: FilesetFileSystem, fileset: FilesetOutput, tmp_path: Path):
        """Test uploading a single file to fileset root path.

        This tests the case: fs.put("local_file.txt", "workspace/fileset")
        which should upload the file to the fileset root.
        """
        base = f"{fileset.workspace}/{fileset.name}"

        # Create a local file
        local_file = tmp_path / "my_file.txt"
        local_file.write_bytes(b"single file content")

        # Test 1: Upload single file to fileset root (no trailing slash on dest)
        # fs.put("my_file.txt", "workspace/fileset") -> workspace/fileset/my_file.txt
        fs.put(str(local_file), base)

        # File should be at fileset root with original name
        assert fs.cat(f"{base}/my_file.txt") == b"single file content"

        # Clean up
        fs.rm(f"{base}/my_file.txt")

        # Test 2: Upload single file to fileset root WITH trailing slash on dest
        # fs.put("my_file.txt", "workspace/fileset/") -> workspace/fileset/my_file.txt
        fs.put(str(local_file), f"{base}/")

        # File should be at fileset root with original name
        assert fs.cat(f"{base}/my_file.txt") == b"single file content"

        # Clean up
        fs.rm(f"{base}/my_file.txt")

        # Test 3: Upload single file with explicit remote filename
        # fs.put("my_file.txt", "workspace/fileset/renamed.txt") -> workspace/fileset/renamed.txt
        fs.put(str(local_file), f"{base}/renamed.txt")

        # File should be renamed
        assert fs.cat(f"{base}/renamed.txt") == b"single file content"

    def test_put_file_requires_file_path(self, fs: FilesetFileSystem, fileset: FilesetOutput, tmp_path: Path):
        """Test that put_file (not put) requires a file path in rpath.

        The low-level put_file method requires rpath to include a file path.
        Calling put_file with just workspace/fileset (no file path) raises ValueError.

        Users should use put() which handles path construction automatically.
        """
        base = f"{fileset.workspace}/{fileset.name}"

        local_file = tmp_path / "test.txt"
        local_file.write_bytes(b"content")

        # put_file with just fileset path (no file path) should raise ValueError
        with pytest.raises(ValueError, match="File path required"):
            fs.put_file(str(local_file), base)

        # put_file with full path works
        fs.put_file(str(local_file), f"{base}/test.txt")
        assert fs.cat(f"{base}/test.txt") == b"content"

    def test_put_trailing_slash_semantics(self, fs: FilesetFileSystem, fileset: FilesetOutput, tmp_path: Path):
        """Test trailing slash semantics for put() per fsspec docs.

        Source trailing slash controls whether to preserve source directory name:
        - "local_dir/" (with slash) -> upload CONTENTS directly to remote
        - "local_dir" (no slash) -> upload directory ITSELF (preserve name)
        """
        base = f"{fileset.workspace}/{fileset.name}"

        # Create local directory structure
        local_dir = tmp_path / "local_data"
        local_dir.mkdir()
        (local_dir / "file1.txt").write_bytes(b"file1")
        (local_dir / "file2.txt").write_bytes(b"file2")
        (local_dir / "nested").mkdir()
        (local_dir / "nested" / "deep.txt").write_bytes(b"deep")

        # Test 1: Source WITH trailing slash -> upload CONTENTS directly
        fs.put(str(local_dir) + "/", f"{base}/upload_contents/", recursive=True)

        # Files should be directly under upload_contents/
        assert fs.cat(f"{base}/upload_contents/file1.txt") == b"file1"
        assert fs.cat(f"{base}/upload_contents/file2.txt") == b"file2"
        assert fs.cat(f"{base}/upload_contents/nested/deep.txt") == b"deep"

        # Test 2: Source WITHOUT trailing slash -> preserve local dir name
        fs.put(str(local_dir), f"{base}/upload_preserve/", recursive=True)

        # local_data/ should be preserved as a subdirectory
        assert fs.cat(f"{base}/upload_preserve/local_data/file1.txt") == b"file1"
        assert fs.cat(f"{base}/upload_preserve/local_data/file2.txt") == b"file2"
        assert fs.cat(f"{base}/upload_preserve/local_data/nested/deep.txt") == b"deep"

    def test_download_entire_fileset(
        self,
        fs: FilesetFileSystem,
        fileset: FilesetOutput,
        sample_dataset: Path,
        tmp_path: Path,
    ):
        """Test uploading and downloading an entire fileset.

        This demonstrates the full round-trip. For fileset root downloads, contents
        are always copied directly. Trailing slash on source only matters for
        subdirectories within a fileset.

        WITH trailing slash (copy contents):
        - put("local_dir/", "remote/") → uploads CONTENTS of local_dir to remote
        - get("remote/", "local_dir/") → downloads CONTENTS of remote to local_dir

        For fileset ROOT (special case):
        - get("workspace/fileset", "local/") → copies contents directly to local/
        - To create a fileset subfolder: get(..., local_path="local/my-fileset/")
        """
        base = f"{fileset.workspace}/{fileset.name}"

        # --- Test WITH trailing slashes (copy contents) ---
        fs.put(str(sample_dataset) + "/", base + "/", recursive=True)

        # Verify files were uploaded at fileset root (not under my_dataset/)
        entries = fs.ls(base, detail=True)
        file_count = sum(1 for e in entries if e["type"] == "file")
        dir_count = sum(1 for e in entries if e["type"] == "directory")
        assert file_count == 1  # README.md
        assert dir_count == 2  # data/, config/

        # Download WITH trailing slashes (copy contents)
        download_with_slash = tmp_path / "with_slash"
        download_with_slash.mkdir()
        fs.get(base + "/", str(download_with_slash) + "/", recursive=True)

        # Verify contents copied directly (no extra nesting)
        assert (download_with_slash / "README.md").read_bytes() == b"# My Dataset"
        assert (download_with_slash / "data" / "train.csv").read_bytes() == b"id,value\n1,100\n2,200"
        assert (download_with_slash / "data" / "test.csv").read_bytes() == b"id,value\n3,300"
        assert (download_with_slash / "config" / "settings.json").read_bytes() == b'{"batch_size": 32}'

        # --- Test WITHOUT trailing slashes (fileset root special case) ---
        # Note: For fileset root, trailing slash doesn't matter - contents are always
        # copied directly. Users who want a subfolder can include the fileset name
        # in local_path.
        download_without_slash = tmp_path / "without_slash"
        download_without_slash.mkdir()
        fs.get(base, str(download_without_slash), recursive=True)

        # Contents go directly into dest (no fileset subfolder)
        assert (download_without_slash / "README.md").read_bytes() == b"# My Dataset"
        assert (download_without_slash / "data" / "train.csv").read_bytes() == b"id,value\n1,100\n2,200"
        assert (download_without_slash / "data" / "test.csv").read_bytes() == b"id,value\n3,300"
        assert (download_without_slash / "config" / "settings.json").read_bytes() == b'{"batch_size": 32}'
        assert not (download_without_slash / fileset.name).exists()

    def test_concurrent_download_failure_hang(self, fs: FilesetFileSystem, fileset: FilesetOutput, tmp_path: Path):
        """Test that a failure in one concurrent download doesn't cause a hang (sync)."""
        from unittest.mock import patch

        # Upload some files
        base = f"{fileset.workspace}/{fileset.name}"
        fs.pipe_file(f"{base}/file1.txt", b"content1")
        fs.pipe_file(f"{base}/file2.txt", b"content2")
        fs.pipe_file(f"{base}/file3.txt", b"content3")

        download_dir = tmp_path / "download"
        download_dir.mkdir()

        call_count = 0
        original_get_file = fs.get_file

        def failing_get_file(rpath, lpath, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 2:  # Fail on the second file
                raise RuntimeError("Simulated failure in concurrent download")
            return original_get_file(rpath, lpath, **kwargs)

        with patch.object(fs, "get_file", side_effect=failing_get_file):
            # _run_coros_in_chunks re-raises the first exception for fsspec compatibility
            with pytest.raises(RuntimeError, match="Simulated failure"):
                fs.get(base + "/", str(download_dir) + "/", recursive=True)

        # If we get here without hanging, the test passes

    def test_concurrent_upload_failure_hang(self, fs: FilesetFileSystem, fileset: FilesetOutput, tmp_path: Path):
        """Test that a failure in one concurrent upload doesn't cause a hang (sync)."""
        from unittest.mock import patch

        # Create local files to upload
        upload_dir = tmp_path / "upload"
        upload_dir.mkdir()
        (upload_dir / "file1.txt").write_bytes(b"content1")
        (upload_dir / "file2.txt").write_bytes(b"content2")
        (upload_dir / "file3.txt").write_bytes(b"content3")

        base = f"{fileset.workspace}/{fileset.name}"

        call_count = 0
        original_put_file = fs.put_file

        def failing_put_file(lpath, rpath, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 2:  # Fail on the second file
                raise RuntimeError("Simulated failure in concurrent upload")
            return original_put_file(lpath, rpath, **kwargs)

        with patch.object(fs, "put_file", side_effect=failing_put_file):
            with pytest.raises(RuntimeError, match="Simulated failure"):
                fs.put(str(upload_dir) + "/", base + "/", recursive=True)

        # If we get here without hanging, the test passes

    def test_get_callback_hooks(self, fs: FilesetFileSystem, fileset: FilesetOutput, tmp_path: Path):
        """Test that get() properly calls callback hooks for progress tracking (sync)."""
        from fsspec.callbacks import Callback

        # Upload some files
        base = f"{fileset.workspace}/{fileset.name}"
        fs.pipe_file(f"{base}/file1.txt", b"content1")
        fs.pipe_file(f"{base}/file2.txt", b"content2")
        fs.pipe_file(f"{base}/file3.txt", b"content3")

        download_dir = tmp_path / "download"
        download_dir.mkdir()

        # Track callback invocations
        hook_calls: list[tuple[int | None, int]] = []
        downloaded_files: list[tuple[str, str]] = []

        # Custom callback that tracks per-file downloads
        class FileTrackingCallback(Callback):
            def branch(self, path_1, path_2, kwargs):
                """Called for each file with source and destination paths."""
                downloaded_files.append((path_1, path_2))
                return super().branch(path_1, path_2, kwargs)

        def progress_hook(size, value, **kwargs):
            hook_calls.append((size, value))

        callback = FileTrackingCallback(hooks={"progress": progress_hook})

        # Download with callback
        fs.get(base + "/", str(download_dir) + "/", recursive=True, callback=callback)

        # Verify set_size was called (first call sets size)
        assert len(hook_calls) >= 1
        assert hook_calls[0][0] == 3  # size should be 3 files

        # Verify relative_update was called for each file (value increments)
        final_size, final_value = hook_calls[-1]
        assert final_size == 3
        assert final_value == 3  # All 3 files downloaded

        # Verify branch() was called for each file with correct paths
        assert len(downloaded_files) == 3
        source_files = {src for src, _ in downloaded_files}
        assert f"{base}#file1.txt" in source_files
        assert f"{base}#file2.txt" in source_files
        assert f"{base}#file3.txt" in source_files

        # Verify files were actually downloaded
        assert (download_dir / "file1.txt").read_bytes() == b"content1"
        assert (download_dir / "file2.txt").read_bytes() == b"content2"
        assert (download_dir / "file3.txt").read_bytes() == b"content3"


class TestAsyncFilesetFileSystem:
    """Test async fsspec operations via AsyncFilesetFileSystem."""

    @pytest.fixture
    def fs(self, async_files_client: AsyncFilesClient) -> AsyncFilesetFileSystem:
        """Create an AsyncFilesetFileSystem backed by the test SDK."""
        return AsyncFilesetFileSystem(client=async_files_client, skip_instance_cache=True)

    async def test_ls_empty_fileset(self, fs: AsyncFilesetFileSystem, fileset: FilesetOutput):
        """Test listing an empty fileset."""
        path = f"{fileset.workspace}/{fileset.name}"
        result = await fs._ls(path)
        assert result == []

    async def test_ls_with_files(self, fs: AsyncFilesetFileSystem, fileset: FilesetOutput):
        """Test listing a fileset with files."""
        base = f"{fileset.workspace}/{fileset.name}"
        await fs._pipe_file(f"{base}/file1.txt", b"content1")
        await fs._pipe_file(f"{base}/file2.txt", b"content2")

        result = await fs._ls(base, detail=True)

        assert len(result) == 2
        names = {r["name"] for r in result}
        assert f"{base}#file1.txt" in names
        assert f"{base}#file2.txt" in names

    async def test_ls_with_directories(self, fs: AsyncFilesetFileSystem, fileset: FilesetOutput):
        """Test that nested files show as directories in listing."""
        base = f"{fileset.workspace}/{fileset.name}"
        await fs._pipe_file(f"{base}/root.txt", b"root")
        await fs._pipe_file(f"{base}/subdir/nested.txt", b"nested")

        result = await fs._ls(base, detail=True)

        # Should see root.txt as file and subdir as directory
        assert len(result) == 2
        types = {r["name"]: r["type"] for r in result}
        assert types[f"{base}#root.txt"] == "file"
        assert types[f"{base}#subdir"] == "directory"

    async def test_ls_subdirectory(self, fs: AsyncFilesetFileSystem, fileset: FilesetOutput):
        """Test listing files in a subdirectory."""
        base = f"{fileset.workspace}/{fileset.name}"
        await fs._pipe_file(f"{base}/subdir/file1.txt", b"content1")
        await fs._pipe_file(f"{base}/subdir/file2.txt", b"content2")

        result = await fs._ls(f"{base}/subdir", detail=False)

        assert len(result) == 2
        assert f"{base}#subdir/file1.txt" in result
        assert f"{base}#subdir/file2.txt" in result

    async def test_cat_file(self, fs: AsyncFilesetFileSystem, fileset: FilesetOutput):
        """Test reading file content with cat."""
        content = b"Hello, fsspec!"
        path = f"{fileset.workspace}/{fileset.name}/test.txt"
        await fs._pipe_file(path, content)

        result = await fs._cat_file(path)

        assert result == content

    async def test_cat_file_with_range(self, fs: AsyncFilesetFileSystem, fileset: FilesetOutput):
        """Test reading partial file content with byte range."""
        content = b"0123456789ABCDEF"
        path = f"{fileset.workspace}/{fileset.name}/test.txt"
        await fs._pipe_file(path, content)

        result = await fs._cat_file(path, start=4, end=10)

        assert result == b"456789"

    async def test_pipe_file(self, fs: AsyncFilesetFileSystem, fileset: FilesetOutput):
        """Test writing file content with pipe."""
        path = f"{fileset.workspace}/{fileset.name}/piped.txt"
        content = b"Piped content"

        await fs._pipe_file(path, content)

        result = await fs._cat_file(path)
        assert result == content

    async def test_put_file(self, fs: AsyncFilesetFileSystem, fileset: FilesetOutput, tmp_path: Path):
        """Test uploading a local file with _put_file."""
        # Create a local file
        local_file = tmp_path / "upload.txt"
        content = b"Content to upload via _put_file"
        local_file.write_bytes(content)

        # Upload to fileset
        remote_path = f"{fileset.workspace}/{fileset.name}/uploaded.txt"
        await fs._put_file(str(local_file), remote_path)

        # Verify content was uploaded
        result = await fs._cat_file(remote_path)
        assert result == content

    async def test_put_file_nested_path(self, fs: AsyncFilesetFileSystem, fileset: FilesetOutput, tmp_path: Path):
        """Test uploading a file to a nested path with _put_file."""
        # Create a local file
        local_file = tmp_path / "nested_upload.txt"
        content = b"Nested upload content"
        local_file.write_bytes(content)

        # Upload to nested path in fileset
        remote_path = f"{fileset.workspace}/{fileset.name}/subdir/nested/uploaded.txt"
        await fs._put_file(str(local_file), remote_path)

        # Verify content was uploaded
        result = await fs._cat_file(remote_path)
        assert result == content

        # Verify parent directories are emulated
        parent_info = await fs._info(f"{fileset.workspace}/{fileset.name}/subdir/nested")
        assert parent_info["type"] == "directory"

    async def test_rm_file(self, fs: AsyncFilesetFileSystem, fileset: FilesetOutput):
        """Test deleting a file."""
        path = f"{fileset.workspace}/{fileset.name}/to_delete.txt"
        await fs._pipe_file(path, b"delete me")

        info = await fs._info(path)
        assert info["type"] == "file"

        await fs._rm_file(path)

        with pytest.raises(FileNotFoundError):
            await fs._info(path)

    async def test_info(self, fs: AsyncFilesetFileSystem, fileset: FilesetOutput):
        """Test getting file info."""
        content = b"Content for info test"
        path = f"{fileset.workspace}/{fileset.name}/info.txt"
        await fs._pipe_file(path, content)

        info = await fs._info(path)

        # info["name"] uses canonical format with # separator
        expected_name = f"{fileset.workspace}/{fileset.name}#info.txt"
        assert info["name"] == expected_name
        assert info["size"] == len(content)
        assert info["type"] == "file"

    async def test_info_directory(self, fs: AsyncFilesetFileSystem, fileset: FilesetOutput):
        """Test getting info for fileset root (directory)."""
        path = f"{fileset.workspace}/{fileset.name}"
        info = await fs._info(path)

        assert info["type"] == "directory"

    async def test_protocol_url(self, fs: AsyncFilesetFileSystem, fileset: FilesetOutput):
        """Test that protocol prefix is handled correctly."""
        content = b"Protocol test"
        base = f"{fileset.workspace}/{fileset.name}"
        await fs._pipe_file(f"{base}/protocol.txt", content)

        # Both with and without protocol should work
        path_no_proto = f"{base}/protocol.txt"
        path_with_proto = f"fileset://{base}/protocol.txt"

        assert await fs._cat_file(path_no_proto) == content
        assert await fs._cat_file(path_with_proto) == content

    async def test_find(self, fs: AsyncFilesetFileSystem, fileset: FilesetOutput):
        """Test recursive file discovery with find (async)."""
        base = f"{fileset.workspace}/{fileset.name}"
        await fs._pipe_file(f"{base}/root.txt", b"root")
        await fs._pipe_file(f"{base}/dir1/file1.txt", b"file1")
        await fs._pipe_file(f"{base}/dir1/file2.txt", b"file2")
        await fs._pipe_file(f"{base}/dir1/subdir/nested.txt", b"nested")
        await fs._pipe_file(f"{base}/dir2/other.txt", b"other")

        # Find all files recursively
        result = await fs._find(base)

        assert len(result) == 5
        assert f"{base}#root.txt" in result
        assert f"{base}#dir1/file1.txt" in result
        assert f"{base}#dir1/subdir/nested.txt" in result

    async def test_glob(self, fs: AsyncFilesetFileSystem, fileset: FilesetOutput):
        """Test pattern matching with glob (async)."""
        base = f"{fileset.workspace}/{fileset.name}"
        await fs._pipe_file(f"{base}/data.csv", b"csv")
        await fs._pipe_file(f"{base}/data.json", b"json")
        await fs._pipe_file(f"{base}/config.json", b"config")
        await fs._pipe_file(f"{base}/subdir/nested.json", b"nested")

        # Use find() to get all files, then filter by extension
        # (glob pattern matching with custom path format is complex due to fnmatch requirements)
        all_files = await fs._find(base)
        json_at_root = [f for f in all_files if f.endswith(".json") and "/" not in f.split("#")[-1]]
        json_all = [f for f in all_files if f.endswith(".json")]

        assert len(json_at_root) == 2
        assert f"{base}#data.json" in json_at_root
        assert f"{base}#config.json" in json_at_root

        # All json files including nested
        assert len(json_all) == 3

    async def test_isdir_isfile(self, fs: AsyncFilesetFileSystem, fileset: FilesetOutput):
        """Test isdir and isfile type checking (async)."""
        base = f"{fileset.workspace}/{fileset.name}"
        await fs._pipe_file(f"{base}/file.txt", b"content")
        await fs._pipe_file(f"{base}/subdir/nested.txt", b"nested")

        # File checks
        info = await fs._info(f"{base}/file.txt")
        assert info["type"] == "file"

        # Directory checks (fileset root)
        assert await fs._isdir(base)

        # Subdirectory checks
        assert await fs._isdir(f"{base}/subdir")

    async def test_cat_multiple_files(self, fs: AsyncFilesetFileSystem, fileset: FilesetOutput):
        """Test reading multiple files at once with cat (async)."""
        base = f"{fileset.workspace}/{fileset.name}"
        await fs._pipe_file(f"{base}/file1.txt", b"content1")
        await fs._pipe_file(f"{base}/file2.txt", b"content2")
        await fs._pipe_file(f"{base}/file3.txt", b"content3")

        # Read multiple files using _cat (async batch read) with new path format
        paths = [f"{base}#file1.txt", f"{base}#file2.txt", f"{base}#file3.txt"]
        result = await fs._cat(paths)

        assert isinstance(result, dict)
        assert result[f"{base}#file1.txt"] == b"content1"
        assert result[f"{base}#file2.txt"] == b"content2"
        assert result[f"{base}#file3.txt"] == b"content3"

    async def test_head_tail(self, fs: AsyncFilesetFileSystem, fileset: FilesetOutput):
        """Test reading first/last bytes of a file (async)."""
        content = b"0123456789ABCDEFGHIJ"
        path = f"{fileset.workspace}/{fileset.name}/test.txt"
        await fs._pipe_file(path, content)

        # Read first 5 bytes using range request
        result = await fs._cat_file(path, start=0, end=5)
        assert result == b"01234"

        # Read last 5 bytes using range request
        result = await fs._cat_file(path, start=15, end=20)
        assert result == b"FGHIJ"

    async def test_get_single_file(self, fs: AsyncFilesetFileSystem, fileset: FilesetOutput, tmp_path: Path):
        """Test downloading a single file with _get() (async version).

        Tests three behaviors:
        - lpath="foo" (no slash, doesn't exist): creates a file named "foo"
        - lpath="foo/" (with slash): creates directory "foo" and puts file inside
        - lpath="existing_dir" (no slash, exists): puts file inside existing directory
        """
        content = b"Single file content"
        base = f"{fileset.workspace}/{fileset.name}"
        await fs._pipe_file(f"{base}/data/test.txt", content)

        # Test 1: lpath without trailing slash - creates file named "renamed"
        renamed_file = tmp_path / "renamed"
        await fs._get(f"{base}/data/test.txt", str(renamed_file))
        assert renamed_file.is_file()
        assert renamed_file.read_bytes() == content

        # Test 2: lpath with trailing slash - creates directory and preserves filename
        target_dir = tmp_path / "target_dir"
        target_dir.mkdir()
        await fs._get(f"{base}/data/test.txt", str(target_dir) + "/")
        assert (target_dir / "test.txt").is_file()
        assert (target_dir / "test.txt").read_bytes() == content

        # Test 3: lpath is existing directory (no slash) - puts file inside
        existing_dir = tmp_path / "existing_dir"
        existing_dir.mkdir()
        await fs._get(f"{base}/data/test.txt", str(existing_dir))
        assert (existing_dir / "test.txt").is_file()
        assert (existing_dir / "test.txt").read_bytes() == content

    async def test_download_entire_fileset(
        self,
        fs: AsyncFilesetFileSystem,
        fileset: FilesetOutput,
        sample_dataset: Path,
        tmp_path: Path,
    ):
        """Test uploading and downloading an entire fileset (async version).

        This demonstrates the full round-trip. For fileset root downloads, contents
        are always copied directly. Trailing slash on source only matters for
        subdirectories within a fileset.

        WITH trailing slash (copy contents):
        - put("local_dir/", "remote/") -> uploads CONTENTS of local_dir to remote
        - get("remote/", "local_dir/") -> downloads CONTENTS of remote to local_dir

        For fileset ROOT (special case):
        - get("workspace/fileset", "local/") -> copies contents directly to local/
        - To create a fileset subfolder: get(..., local_path="local/my-fileset/")
        """
        base = f"{fileset.workspace}/{fileset.name}"

        # --- Test WITH trailing slashes (copy contents) ---
        await fs._put(str(sample_dataset) + "/", base + "/", recursive=True)

        # Verify files were uploaded at fileset root (not under my_dataset/)
        entries = await fs._ls(base, detail=True)
        file_count = sum(1 for e in entries if e["type"] == "file")
        dir_count = sum(1 for e in entries if e["type"] == "directory")
        assert file_count == 1  # README.md
        assert dir_count == 2  # data/, config/

        # Download WITH trailing slashes (copy contents)
        download_with_slash = tmp_path / "with_slash"
        download_with_slash.mkdir()
        await fs._get(base + "/", str(download_with_slash) + "/", recursive=True)

        # Verify contents copied directly (no extra nesting)
        assert (download_with_slash / "README.md").read_bytes() == b"# My Dataset"
        assert (download_with_slash / "data" / "train.csv").read_bytes() == b"id,value\n1,100\n2,200"
        assert (download_with_slash / "data" / "test.csv").read_bytes() == b"id,value\n3,300"
        assert (download_with_slash / "config" / "settings.json").read_bytes() == b'{"batch_size": 32}'

        # --- Test WITHOUT trailing slashes (fileset root special case) ---
        # Note: For fileset root, trailing slash doesn't matter - contents are always
        # copied directly. Users who want a subfolder can include the fileset name
        # in local_path.
        download_without_slash = tmp_path / "without_slash"
        download_without_slash.mkdir()
        await fs._get(base, str(download_without_slash), recursive=True)

        # Contents go directly into dest (no fileset subfolder)
        assert (download_without_slash / "README.md").read_bytes() == b"# My Dataset"
        assert (download_without_slash / "data" / "train.csv").read_bytes() == b"id,value\n1,100\n2,200"
        assert (download_without_slash / "data" / "test.csv").read_bytes() == b"id,value\n3,300"
        assert (download_without_slash / "config" / "settings.json").read_bytes() == b'{"batch_size": 32}'
        assert not (download_without_slash / fileset.name).exists()

    async def test_concurrent_download_failure_hang(
        self, fs: AsyncFilesetFileSystem, fileset: FilesetOutput, tmp_path: Path
    ):
        """Test that a failure in one concurrent download doesn't cause a hang.

        This reproduces an issue where orphaned asyncio tasks from failed concurrent
        downloads keep AnyIO worker threads (non-daemon) alive, preventing process exit.
        The fix uses anyio TaskGroup which properly cancels sibling tasks on failure.
        """
        from unittest.mock import patch

        # Upload some files
        base = f"{fileset.workspace}/{fileset.name}"
        await fs._pipe_file(f"{base}/file1.txt", b"content1")
        await fs._pipe_file(f"{base}/file2.txt", b"content2")
        await fs._pipe_file(f"{base}/file3.txt", b"content3")

        download_dir = tmp_path / "download"
        download_dir.mkdir()

        call_count = 0
        original_get_file = fs._get_file

        async def failing_get_file(rpath, lpath, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 2:  # Fail on the second file
                raise RuntimeError("Simulated failure in concurrent download")
            return await original_get_file(rpath, lpath, **kwargs)

        with patch.object(fs, "_get_file", side_effect=failing_get_file):
            # _run_coros_in_chunks re-raises the first exception for fsspec compatibility
            with pytest.raises(RuntimeError, match="Simulated failure"):
                await fs._get(base + "/", str(download_dir) + "/", recursive=True)

        # If we get here without hanging, the test passes

    async def test_concurrent_upload_failure_hang(
        self, fs: AsyncFilesetFileSystem, fileset: FilesetOutput, tmp_path: Path
    ):
        """Test that a failure in one concurrent upload doesn't cause a hang.

        Similar to test_concurrent_download_failure_hang, this ensures that the
        _put implementation properly cancels sibling tasks on failure.
        """
        from unittest.mock import patch

        # Create local files to upload
        upload_dir = tmp_path / "upload"
        upload_dir.mkdir()
        (upload_dir / "file1.txt").write_bytes(b"content1")
        (upload_dir / "file2.txt").write_bytes(b"content2")
        (upload_dir / "file3.txt").write_bytes(b"content3")

        base = f"{fileset.workspace}/{fileset.name}"

        call_count = 0
        original_put_file = fs._put_file

        async def failing_put_file(lpath, rpath, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 2:  # Fail on the second file
                raise RuntimeError("Simulated failure in concurrent upload")
            return await original_put_file(lpath, rpath, **kwargs)

        with patch.object(fs, "_put_file", side_effect=failing_put_file):
            with pytest.raises(RuntimeError, match="Simulated failure"):
                await fs._put(str(upload_dir) + "/", base + "/", recursive=True)

        # If we get here without hanging, the test passes

    async def test_batch_size_limits_concurrency(
        self,
        async_files_client: AsyncFilesClient,
        fileset: FilesetOutput,
        tmp_path: Path,
    ):
        """Test that batch_size properly limits concurrent operations.

        Creates a filesystem with batch_size=4, then downloads 8 files and verifies
        that no more than 4 downloads run concurrently at any time.
        """

        batch_size = 4
        total_files = 8

        # Create filesystem with limited concurrency
        fs = AsyncFilesetFileSystem(client=async_files_client, batch_size=batch_size)

        # Upload files
        base = f"{fileset.workspace}/{fileset.name}"
        for i in range(total_files):
            await fs._pipe_file(f"{base}/file{i}.txt", f"content{i}".encode())

        download_dir = tmp_path / "download"
        download_dir.mkdir()

        # Synchronization primitives
        started_count = 0
        batch_started = anyio.Event()  # Signals when batch_size tasks have started
        proceed = anyio.Event()  # Gate to release tasks
        lock = anyio.Lock()

        original_get_file = fs._get_file

        async def tracking_get_file(rpath, lpath, **kwargs):
            nonlocal started_count

            async with lock:
                started_count += 1
                if started_count == batch_size:
                    batch_started.set()

            # Wait at gate until test releases us
            await proceed.wait()
            return await original_get_file(rpath, lpath, **kwargs)

        async def run_download():
            with patch.object(fs, "_get_file", side_effect=tracking_get_file):
                await fs._get(base + "/", str(download_dir) + "/", recursive=True)

        async with anyio.create_task_group() as tg:
            tg.start_soon(run_download)

            # Wait until batch_size tasks have started (proves they're running concurrently)
            with anyio.fail_after(5):  # Timeout if something goes wrong
                await batch_started.wait()

            # At this point, exactly batch_size tasks started and are waiting at the gate.
            # If concurrency wasn't limited, all 8 would have started.
            async with lock:
                assert started_count == batch_size, (
                    f"Expected exactly {batch_size} concurrent tasks, got {started_count}"
                )

            # Release the gate to let all tasks complete
            proceed.set()

        # Verify all files were downloaded
        downloaded_files = list(download_dir.iterdir())
        assert len(downloaded_files) == total_files

    async def test_get_callback_hooks(self, fs: AsyncFilesetFileSystem, fileset: FilesetOutput, tmp_path: Path):
        """Test that _get properly calls callback hooks for progress tracking.

        This test demonstrates two callback features:
        1. Progress hooks that track (size, value) for overall progress
        2. Per-file tracking via branch() which receives source/dest paths
        """
        from fsspec.callbacks import Callback

        # Upload some files
        base = f"{fileset.workspace}/{fileset.name}"
        await fs._pipe_file(f"{base}/file1.txt", b"content1")
        await fs._pipe_file(f"{base}/file2.txt", b"content2")
        await fs._pipe_file(f"{base}/file3.txt", b"content3")

        download_dir = tmp_path / "download"
        download_dir.mkdir()

        # Track callback invocations
        hook_calls: list[tuple[int | None, int]] = []
        downloaded_files: list[tuple[str, str]] = []

        # Custom callback that tracks per-file downloads
        class FileTrackingCallback(Callback):
            def branch(self, path_1, path_2, kwargs):
                """Called for each file with source and destination paths."""
                downloaded_files.append((path_1, path_2))
                return super().branch(path_1, path_2, kwargs)

        def progress_hook(size, value, **kwargs):
            hook_calls.append((size, value))

        callback = FileTrackingCallback(hooks={"progress": progress_hook})

        # Download with callback
        await fs._get(base + "/", str(download_dir) + "/", recursive=True, callback=callback)

        # Verify set_size was called (first call sets size)
        assert len(hook_calls) >= 1
        assert hook_calls[0][0] == 3  # size should be 3 files

        # Verify relative_update was called for each file (value increments)
        final_size, final_value = hook_calls[-1]
        assert final_size == 3
        assert final_value == 3  # All 3 files downloaded

        # Verify branch() was called for each file with correct paths
        assert len(downloaded_files) == 3
        source_files = {src for src, _ in downloaded_files}
        assert f"{base}#file1.txt" in source_files
        assert f"{base}#file2.txt" in source_files
        assert f"{base}#file3.txt" in source_files

        # Verify destination paths point to download directory
        for _, dest in downloaded_files:
            assert dest.startswith(str(download_dir))

        # Verify files were actually downloaded
        assert (download_dir / "file1.txt").read_bytes() == b"content1"
        assert (download_dir / "file2.txt").read_bytes() == b"content2"
        assert (download_dir / "file3.txt").read_bytes() == b"content3"

    async def test_get_per_chunk_callbacks(self, fs: AsyncFilesetFileSystem, fileset: FilesetOutput, tmp_path: Path):
        """Test that _get passes branched callbacks to _get_file for per-chunk progress.

        This test verifies the full callback hierarchy:
        1. Parent callback tracks overall file count progress
        2. branch() creates child callbacks for each file
        3. Child callbacks receive per-chunk byte progress during download
        """
        from fsspec.callbacks import Callback

        # Upload a file with enough content to have multiple chunks
        base = f"{fileset.workspace}/{fileset.name}"
        # Create content large enough to potentially have multiple chunks
        large_content = b"x" * 10000
        await fs._pipe_file(f"{base}/large_file.bin", large_content)

        download_dir = tmp_path / "download"
        download_dir.mkdir()

        # Track all callback activity
        parent_calls: list[tuple[str, int | None, int]] = []
        child_calls: list[tuple[str, str, int | None, int]] = []

        class ChunkTrackingCallback(Callback):
            """Parent callback that creates per-file child callbacks."""

            def set_size(self, size):
                parent_calls.append(("set_size", size, 0))
                super().set_size(size)

            def relative_update(self, inc=1):
                super().relative_update(inc)
                parent_calls.append(("relative_update", self.size, self.value))

            def branched(self, path_1, path_2, **kwargs):
                """Return a child callback that tracks per-chunk progress."""
                parent_calls.append(("branched", None, 0))
                return PerFileCallback(path_1, path_2)

        class PerFileCallback(Callback):
            """Child callback for tracking per-chunk progress within a single file."""

            def __init__(self, src: str, dst: str):
                super().__init__()
                self.src = src
                self.dst = dst

            def set_size(self, size):
                child_calls.append(("set_size", self.src, size, 0))
                super().set_size(size)

            def relative_update(self, inc=1):
                super().relative_update(inc)
                child_calls.append(("relative_update", self.src, self.size, self.value))

        callback = ChunkTrackingCallback()

        # Download with callback
        await fs._get(base + "/", str(download_dir) + "/", recursive=True, callback=callback)

        # Verify parent callback was used for file-level progress
        parent_set_size = [c for c in parent_calls if c[0] == "set_size"]
        assert len(parent_set_size) >= 1, "Parent set_size should be called"
        assert parent_set_size[0][1] == 1, "Should have 1 file"

        parent_updates = [c for c in parent_calls if c[0] == "relative_update"]
        assert len(parent_updates) >= 1, "Parent should get file completion updates"

        parent_branches = [c for c in parent_calls if c[0] == "branched"]
        assert len(parent_branches) == 1, "branched() should be called once per file"

        # Verify child callback received per-chunk updates
        # Note: set_size may not be called if server doesn't return Content-Length header
        child_set_size = [c for c in child_calls if c[0] == "set_size"]
        if child_set_size:
            # If Content-Length was present, verify it was correct
            assert child_set_size[0][2] == len(large_content), f"Child size should be {len(large_content)}"

        # Child should have relative_update called (potentially multiple times for chunks)
        child_updates = [c for c in child_calls if c[0] == "relative_update"]
        assert len(child_updates) >= 1, "Child should receive chunk progress updates"

        # Final child update should show all bytes downloaded
        final_child_update = child_updates[-1]
        assert final_child_update[3] == len(large_content), "Final update should show all bytes"

        # Verify file was actually downloaded correctly
        assert (download_dir / "large_file.bin").read_bytes() == large_content

    async def test_put_per_chunk_callbacks(self, fs: AsyncFilesetFileSystem, fileset: FilesetOutput, tmp_path: Path):
        """Test that _put passes branched callbacks to _put_file for per-chunk progress.

        This test verifies the full callback hierarchy for uploads:
        1. Parent callback tracks overall file count progress
        2. branch() creates child callbacks for each file
        3. Child callbacks receive per-chunk byte progress during upload
        """
        from fsspec.callbacks import Callback

        # Create a local file with enough content to have multiple chunks
        large_content = b"y" * 10000
        local_file = tmp_path / "upload_file.bin"
        local_file.write_bytes(large_content)

        base = f"{fileset.workspace}/{fileset.name}"

        # Track all callback activity
        parent_calls: list[tuple[str, int | None, int]] = []
        child_calls: list[tuple[str, str, int | None, int]] = []

        class ChunkTrackingCallback(Callback):
            """Parent callback that creates per-file child callbacks."""

            def set_size(self, size):
                parent_calls.append(("set_size", size, 0))
                super().set_size(size)

            def relative_update(self, inc=1):
                super().relative_update(inc)
                parent_calls.append(("relative_update", self.size, self.value))

            def branched(self, path_1, path_2, **kwargs):
                """Return a child callback context manager that tracks per-chunk progress."""
                parent_calls.append(("branched", None, 0))
                return PerFileCallback(path_1, path_2)

        class PerFileCallback(Callback):
            """Child callback for tracking per-chunk progress within a single file."""

            def __init__(self, src: str, dst: str):
                super().__init__()
                self.src = src
                self.dst = dst

            def __enter__(self):
                return self

            def __exit__(self, *args):
                pass

            def set_size(self, size):
                child_calls.append(("set_size", self.src, size, 0))
                super().set_size(size)

            def relative_update(self, inc=1):
                super().relative_update(inc)
                child_calls.append(("relative_update", self.src, self.size, self.value))

        callback = ChunkTrackingCallback()

        # Upload with callback using put() which calls _put_file internally
        await fs._put(str(local_file), f"{base}/uploaded.bin", callback=callback)

        # Verify parent callback was used for file-level progress
        parent_set_size = [c for c in parent_calls if c[0] == "set_size"]
        assert len(parent_set_size) >= 1, "Parent set_size should be called"
        assert parent_set_size[0][1] == 1, "Should have 1 file"

        parent_updates = [c for c in parent_calls if c[0] == "relative_update"]
        assert len(parent_updates) >= 1, "Parent should get file completion updates"

        parent_branches = [c for c in parent_calls if c[0] == "branched"]
        assert len(parent_branches) == 1, "branched() should be called once per file"

        # Verify child callback received per-chunk updates
        child_set_size = [c for c in child_calls if c[0] == "set_size"]
        assert len(child_set_size) >= 1, "Child set_size should be called with file size"
        assert child_set_size[0][2] == len(large_content), f"Child size should be {len(large_content)}"

        # Child should have relative_update called (potentially multiple times for chunks)
        child_updates = [c for c in child_calls if c[0] == "relative_update"]
        assert len(child_updates) >= 1, "Child should receive chunk progress updates"

        # Final child update should show all bytes uploaded
        final_child_update = child_updates[-1]
        assert final_child_update[3] == len(large_content), "Final update should show all bytes"

        # Verify file was actually uploaded correctly
        downloaded = await fs._cat_file(f"{base}/uploaded.bin")
        assert downloaded == large_content


class TestDuckDBIntegration:
    """Test DuckDB queries via fileset:// fsspec protocol.

    These tests verify that DuckDB can query parquet files stored in filesets
    using the fileset:// URL scheme. DuckDB discovers the filesystem via fsspec's
    protocol registry and creates it automatically.
    """

    def test_duckdb_parquet_query(
        self,
        sdk: NeMoPlatform,
        fileset: FilesetOutput,
    ):
        """Test querying a parquet file with DuckDB via fileset:// protocol."""
        # Create test data
        df = pd.DataFrame(
            {
                "id": range(1, 101),
                "name": [f"item_{i}" for i in range(1, 101)],
                "value": [i * 10.5 for i in range(1, 101)],
                "category": ["A" if i % 2 == 0 else "B" for i in range(1, 101)],
            }
        )
        parquet_bytes = df.to_parquet(index=False)

        file_path = "data.parquet"

        # Create filesystem via fsspec (how users would configure it)
        fs = fsspec.filesystem(
            "fileset",
            client=client_from_platform(sdk, FilesClient),
        )
        fs.pipe(f"{fileset.workspace}/{fileset.name}#{file_path}", parquet_bytes)

        # Query with DuckDB using fileset:// URL with new # format
        fileset_url = f"fileset://{fileset.workspace}/{fileset.name}#{file_path}"
        conn = duckdb.connect()
        conn.register_filesystem(fs)

        # Simple query
        result = conn.execute(f"SELECT * FROM '{fileset_url}' WHERE id <= 5").fetchdf()
        assert len(result) == 5
        assert list(result["id"]) == [1, 2, 3, 4, 5]

        # Range query with filter
        result = conn.execute(f"SELECT * FROM '{fileset_url}' WHERE value >= 500 AND value <= 600").fetchdf()
        expected_ids = [i for i in range(1, 101) if 500 <= i * 10.5 <= 600]
        assert len(result) == len(expected_ids)

        # Aggregation query
        result = conn.execute(
            f"SELECT category, COUNT(*) as cnt, AVG(value) as avg_val FROM '{fileset_url}' GROUP BY category"
        ).fetchdf()
        assert len(result) == 2
        assert set(result["category"]) == {"A", "B"}

    def test_duckdb_parquet_range_read(
        self,
        sdk: NeMoPlatform,
        fileset: FilesetOutput,
    ):
        """Test that DuckDB performs efficient range reads on parquet files.

        Parquet files store metadata at the end (footer), so DuckDB reads:
        1. The footer first (to get schema and row group info)
        2. Only the relevant row groups/columns based on the query

        This test verifies that our fsspec implementation supports the
        range requests that DuckDB makes for efficient parquet reads.
        """
        # Create a larger dataset to ensure multiple row groups
        df = pd.DataFrame(
            {
                "id": range(1, 10001),
                "data": [f"row_{i}" for i in range(1, 10001)],
                "value": [float(i) for i in range(1, 10001)],
            }
        )
        parquet_bytes = df.to_parquet(index=False)

        file_path = "large_data.parquet"

        # Create filesystem via fsspec
        fs = fsspec.filesystem(
            "fileset",
            client=client_from_platform(sdk, FilesClient),
        )
        fs.pipe(f"{fileset.workspace}/{fileset.name}#{file_path}", parquet_bytes)
        fileset_url = f"fileset://{fileset.workspace}/{fileset.name}#{file_path}"
        conn = duckdb.connect()
        conn.register_filesystem(fs)

        # Query that should only read a small portion of the file
        result = conn.execute(f"SELECT id, value FROM '{fileset_url}' WHERE id BETWEEN 100 AND 110").fetchdf()

        assert len(result) == 11
        assert list(result["id"]) == list(range(100, 111))
        assert list(result["value"]) == [float(i) for i in range(100, 111)]

    def test_duckdb_legacy_path_format(
        self,
        sdk: NeMoPlatform,
        fileset: FilesetOutput,
    ):
        """Test DuckDB queries using legacy workspace/fileset/path format.

        This validates backwards compatibility with the legacy path format
        (without # separator) for DuckDB's fsspec integration.
        """
        # Create test data
        df = pd.DataFrame(
            {
                "id": range(1, 51),
                "name": [f"item_{i}" for i in range(1, 51)],
            }
        )
        parquet_bytes = df.to_parquet(index=False)

        file_path = "test_legacy_format.parquet"

        # Create filesystem via fsspec
        fs = fsspec.filesystem(
            "fileset",
            client=client_from_platform(sdk, FilesClient),
        )
        fs.pipe(f"{fileset.workspace}/{fileset.name}#{file_path}", parquet_bytes)

        # Query with DuckDB using LEGACY path format: workspace/fileset/path
        fileset_url = f"fileset://{fileset.workspace}/{fileset.name}/{file_path}"
        conn = duckdb.connect()
        conn.register_filesystem(fs)

        result = conn.execute(f"SELECT * FROM '{fileset_url}' WHERE id <= 10").fetchdf()
        assert len(result) == 10
        assert list(result["id"]) == list(range(1, 11))


class TestDirCache:
    """Test directory listing caching behavior.

    These tests verify that the dircache is populated correctly and reduces
    API calls when traversing directory structures.
    """

    @pytest.fixture
    def fs(self, sdk: NeMoPlatform) -> FilesetFileSystem:
        """Create a FilesetFileSystem backed by the test SDK."""
        return FilesetFileSystem(client=client_from_platform(sdk, FilesClient))

    def test_ls_populates_cache_for_nested_dirs(self, fs: FilesetFileSystem, fileset: FilesetOutput):
        """_ls should populate cache for all directory levels in the response."""
        base = f"{fileset.workspace}/{fileset.name}"

        # Upload files in nested structure
        fs.pipe(f"{base}/file1.txt", b"root file")
        fs.pipe(f"{base}/subdir/file2.txt", b"nested file")
        fs.pipe(f"{base}/subdir/nested/file3.txt", b"deep nested file")

        # Clear cache and list root
        fs.invalidate_cache()
        fs.ls(base)

        # Cache should contain entries for all directory levels
        # Note: cache keys use # separator: workspace/fileset#subdir
        assert base in fs.dircache
        assert f"{base}#subdir" in fs.dircache
        assert f"{base}#subdir/nested" in fs.dircache

    def test_deeply_nested_tree(self, fs: FilesetFileSystem, fileset: FilesetOutput):
        """Cache should handle deeply nested directory structures."""
        base = f"{fileset.workspace}/{fileset.name}"

        # Create a 6-level deep structure with files at various levels
        fs.pipe(f"{base}/a/b/c/d/e/f/deep.txt", b"deepest file")
        fs.pipe(f"{base}/a/b/c/mid.txt", b"mid-level file")
        fs.pipe(f"{base}/a/shallow.txt", b"shallow file")
        # Add a sibling branch
        fs.pipe(f"{base}/a/b/other/branch.txt", b"branch file")

        fs.invalidate_cache()
        fs.ls(base)

        # All directory levels should be cached (using # separator for paths)
        assert f"{base}#a" in fs.dircache
        assert f"{base}#a/b" in fs.dircache
        assert f"{base}#a/b/c" in fs.dircache
        assert f"{base}#a/b/c/d" in fs.dircache
        assert f"{base}#a/b/c/d/e" in fs.dircache
        assert f"{base}#a/b/c/d/e/f" in fs.dircache
        assert f"{base}#a/b/other" in fs.dircache

        # Verify nested ls calls use cache (no new entries added)
        cache_size = len(fs.dircache)
        fs.ls(f"{base}/a/b/c/d/e/f")
        assert len(fs.dircache) == cache_size

        # Verify file info is correct at various depths
        info = fs.info(f"{base}/a/b/c/d/e/f/deep.txt")
        assert info["type"] == "file"
        info = fs.info(f"{base}/a/b/c/mid.txt")
        assert info["type"] == "file"

    def test_nested_ls_uses_cache(self, fs: FilesetFileSystem, fileset: FilesetOutput):
        """Subsequent _ls calls for nested paths should use cache."""
        base = f"{fileset.workspace}/{fileset.name}"

        # Upload files
        fs.pipe(f"{base}/file1.txt", b"root file")
        fs.pipe(f"{base}/subdir/file2.txt", b"nested file")

        # Clear cache and list root
        fs.invalidate_cache()

        # Count API calls by tracking dircache state
        initial_cache_size = len(fs.dircache)
        fs.ls(base)
        after_root_ls = len(fs.dircache)

        # Nested ls should NOT increase cache size (it's already populated)
        fs.ls(f"{base}/subdir")
        after_nested_ls = len(fs.dircache)

        # Cache was populated by first ls call
        assert after_root_ls > initial_cache_size
        # Nested ls used cache, didn't add new entries
        assert after_nested_ls == after_root_ls

    def test_find_populates_cache(self, fs: FilesetFileSystem, fileset: FilesetOutput):
        """_find should populate cache for subsequent _ls calls."""
        base = f"{fileset.workspace}/{fileset.name}"

        # Upload files
        fs.pipe(f"{base}/file1.txt", b"root file")
        fs.pipe(f"{base}/subdir/file2.txt", b"nested file")

        # Clear cache
        fs.invalidate_cache()

        # find() should populate cache
        fs.find(base)

        # Cache should be populated (using # separator for subdirs)
        assert base in fs.dircache
        assert f"{base}#subdir" in fs.dircache

    def test_info_uses_cache(self, fs: FilesetFileSystem, fileset: FilesetOutput):
        """_info should use dircache instead of making API calls."""
        base = f"{fileset.workspace}/{fileset.name}"

        # Upload files
        fs.pipe(f"{base}/file1.txt", b"content")
        fs.pipe(f"{base}/subdir/file2.txt", b"nested content")

        # Clear cache and populate via ls
        fs.invalidate_cache()
        fs.ls(base)

        # info() calls should use cache - verify they return correct types
        info = fs.info(base)
        assert info["type"] == "directory"

        info = fs.info(f"{base}/subdir")
        assert info["type"] == "directory"

        info = fs.info(f"{base}/file1.txt")
        assert info["type"] == "file"
        assert info["size"] == len(b"content")

    def test_cache_invalidation_on_write(self, fs: FilesetFileSystem, fileset: FilesetOutput):
        """Cache should be invalidated when files are written."""
        base = f"{fileset.workspace}/{fileset.name}"

        # Upload initial file and populate cache
        fs.pipe(f"{base}/file1.txt", b"content1")
        fs.ls(base)
        assert base in fs.dircache

        # Write new file - should invalidate parent cache
        fs.pipe(f"{base}/file2.txt", b"content2")

        # Parent directory cache should be invalidated
        assert base not in fs.dircache

        # New ls should show both files (paths use # separator)
        result = fs.ls(base, detail=False)
        assert f"{base}#file1.txt" in result
        assert f"{base}#file2.txt" in result

    def test_cache_invalidation_on_delete(self, fs: FilesetFileSystem, fileset: FilesetOutput):
        """Cache should be invalidated when files are deleted."""
        base = f"{fileset.workspace}/{fileset.name}"

        # Upload files and populate cache
        fs.pipe(f"{base}/file1.txt", b"content1")
        fs.pipe(f"{base}/file2.txt", b"content2")
        fs.ls(base)
        assert base in fs.dircache

        # Delete file - should invalidate parent cache
        fs.rm(f"{base}/file1.txt")

        # Parent directory cache should be invalidated
        assert base not in fs.dircache

        # New ls should show only remaining file (paths use # separator)
        result = fs.ls(base, detail=False)
        assert f"{base}#file1.txt" not in result
        assert f"{base}#file2.txt" in result

    def test_refresh_bypasses_cache(self, fs: FilesetFileSystem, fileset: FilesetOutput):
        """_ls with refresh=True should bypass cache."""
        base = f"{fileset.workspace}/{fileset.name}"

        # Upload file and populate cache
        fs.pipe(f"{base}/file1.txt", b"content")
        fs.ls(base)

        # Verify cache is populated
        assert base in fs.dircache

        # ls with refresh should still work
        result = fs.ls(base, refresh=True)
        assert len(result) == 1

    def test_info_file_not_found(self, fs: FilesetFileSystem, fileset: FilesetOutput):
        """_info should raise FileNotFoundError for non-existent paths."""
        base = f"{fileset.workspace}/{fileset.name}"

        # Upload a file and populate cache
        fs.pipe(f"{base}/exists.txt", b"content")
        fs.ls(base)

        # info() on non-existent file should raise FileNotFoundError
        with pytest.raises(FileNotFoundError):
            fs.info(f"{base}/nonexistent.txt")

    @pytest.mark.parametrize("detail", [True, False])
    def test_ls_detail_parameter(self, fs: FilesetFileSystem, fileset: FilesetOutput, detail: bool):
        """_ls should respect the detail parameter."""
        base = f"{fileset.workspace}/{fileset.name}"

        # Upload files
        fs.pipe(f"{base}/file1.txt", b"content1")
        fs.pipe(f"{base}/subdir/file2.txt", b"content2")

        result = fs.ls(base, detail=detail)

        if detail:
            assert all(isinstance(item, dict) for item in result)
            assert all("name" in item and "type" in item for item in result)
        else:
            assert all(isinstance(item, str) for item in result)

    def test_cache_disabled(
        self,
        sdk: NeMoPlatform,
        fileset: FilesetOutput,
    ):
        """When use_listings_cache=False, cache should not be used."""
        # Create filesystem with cache disabled
        fs = FilesetFileSystem(client=client_from_platform(sdk, FilesClient))
        fs.dircache.use_listings_cache = False

        base = f"{fileset.workspace}/{fileset.name}"

        # Upload file
        fs.pipe(f"{base}/file1.txt", b"content")

        # Multiple ls calls - cache should remain empty
        fs.ls(base)
        assert base not in fs.dircache

        fs.ls(base)
        assert base not in fs.dircache
