# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import MagicMock

from nemo_platform_plugin.jobs.file_manager import _filter_files_by_patterns
from nmp.common.jobs.file_manager import FilesetFileManager, FileStorageType

# =============================================================================
# Pattern Matching Tests
# =============================================================================


class TestFilterFilesByPatterns:
    """Tests for the _filter_files_by_patterns function."""

    def test_no_patterns_returns_all_files(self):
        files = ["a.txt", "b.pyc", "c/d.txt"]
        assert _filter_files_by_patterns(files, None) == files
        assert _filter_files_by_patterns(files, []) == files

    def test_single_pattern_string(self):
        files = ["a.txt", "b.pyc", "c.pyc"]
        result = _filter_files_by_patterns(files, "*.pyc")
        assert result == ["a.txt"]

    def test_single_pattern_list(self):
        files = ["a.txt", "b.pyc", "c.pyc"]
        result = _filter_files_by_patterns(files, ["*.pyc"])
        assert result == ["a.txt"]

    def test_multiple_patterns(self):
        files = ["a.txt", "b.pyc", "c.log", "d.tmp"]
        result = _filter_files_by_patterns(files, ["*.pyc", "*.log"])
        assert result == ["a.txt", "d.tmp"]

    def test_nested_files_with_extension_pattern(self):
        files = [
            "root.txt",
            "root.pyc",
            "subdir/nested.txt",
            "subdir/nested.pyc",
            "a/b/deep.txt",
            "a/b/deep.pyc",
        ]
        result = _filter_files_by_patterns(files, "*.pyc")
        assert result == ["root.txt", "subdir/nested.txt", "a/b/deep.txt"]

    def test_filename_pattern_matches_at_any_depth(self):
        """Test that filename patterns match at any depth using Path.match()."""
        files = [
            "cache.db",
            "foo/cache.db",
            "foo/bar/cache.db",
            "other.db",
            "foo/other.db",
        ]
        result = _filter_files_by_patterns(files, "cache.db")
        assert result == ["other.db", "foo/other.db"]

    def test_directory_pattern(self):
        """Test that dir/ pattern excludes files inside directories with that name."""
        files = [
            "results.json",
            "cache/model.bin",
            "cache/data.json",
            "foo/cache/model.bin",
            "metrics.json",
        ]
        result = _filter_files_by_patterns(files, "cache/")
        assert result == ["results.json", "metrics.json"]

    def test_combined_patterns_from_evaluator(self):
        """Test the actual patterns used by the evaluator service."""
        files = [
            "aggregate-scores.json",
            "metrics.json",
            "cache.db",
            "subdir/cache.db",
            "cache/model.bin",
            "cache/data/file.txt",
            "foo/cache/model.bin",
        ]
        patterns = ["cache.db", "cache/"]
        result = _filter_files_by_patterns(files, patterns)
        assert result == ["aggregate-scores.json", "metrics.json"]

    def test_empty_file_list(self):
        result = _filter_files_by_patterns([], ["*.pyc"])
        assert result == []

    def test_no_matches(self):
        files = ["a.txt", "b.json", "c.yaml"]
        result = _filter_files_by_patterns(files, "*.pyc")
        assert result == files


# =============================================================================
# FilesetFileManager Tests
# =============================================================================


def test_fileset_url():
    """Test URL generation for fileset storage."""
    filesystem = MagicMock()
    mgr = FilesetFileManager(
        workspace="my-workspace",
        fileset_name="my-fileset",
        filesystem=filesystem,
    )
    assert mgr.url() == "my-workspace/my-fileset"
    assert mgr.url("path/to/file") == "my-workspace/my-fileset#path/to/file"


def test_fileset_storage_type():
    """Test storage type returns FILESET."""
    filesystem = MagicMock()
    mgr = FilesetFileManager(
        workspace="my-workspace",
        fileset_name="my-fileset",
        filesystem=filesystem,
    )
    assert mgr.storage_type() == FileStorageType.FILESET


def test_fileset_validate_storage_exists(fileset_manager, mock_fileset_fs):
    """Test validate_storage when fileset already exists."""
    fileset_manager.validate_storage()
    mock_fileset_fs.info.assert_called()


def test_fileset_validate_storage_creates(fileset_manager, mock_fileset_fs):
    """Test validate_storage creates fileset when missing."""
    mock_fileset_fs._client = MagicMock()
    mock_fileset_fs.info.side_effect = FileNotFoundError("not found")
    fileset_manager.validate_storage()
    mock_fileset_fs._client.create_fileset.assert_called_once()


def test_fileset_upload_file(tmp_path, fileset_manager, mock_fileset_fs):
    """Test uploading a single file."""
    test_file = tmp_path / "test.txt"
    test_file.write_text("content")

    result = fileset_manager.upload(test_file, "remote/test.txt")

    mock_fileset_fs.put_file.assert_called()
    assert result == "default/job-results-jobid-123#remote/test.txt"


def test_fileset_upload_directory(tmp_path, fileset_manager, mock_fileset_fs):
    """Test uploading a directory."""
    test_dir = tmp_path / "mydir"
    test_dir.mkdir()
    (test_dir / "file.txt").write_text("content")

    result = fileset_manager.upload(test_dir, "remote/mydir")

    mock_fileset_fs.put_file.assert_called()
    assert result == "default/job-results-jobid-123#remote/mydir"


def test_fileset_download_from_url_file(tmp_path, fileset_manager, mock_fileset_fs):
    """Test downloading a single file from URL."""
    mock_fileset_fs.info.return_value = {"name": "test", "size": 100, "type": "file"}

    result = fileset_manager.download_from_url(
        url="default/job-results-jobid-123#remote/test.txt",
        local_dir=tmp_path,
    )

    mock_fileset_fs.get_file.assert_called()
    assert result.tmp_dir == tmp_path


def test_fileset_download_from_url_directory(tmp_path, fileset_manager, mock_fileset_fs):
    """Test downloading a directory from URL."""
    mock_fileset_fs.info.return_value = {"name": "test", "size": 0, "type": "directory"}

    result = fileset_manager.download_from_url(
        url="default/job-results-jobid-123#remote/mydir",
        local_dir=tmp_path,
    )

    mock_fileset_fs.get.assert_called()
    assert result.tmp_dir == tmp_path


def test_fileset_download_from_legacy_url_normalizes_to_hash_file_ref(tmp_path, fileset_manager, mock_fileset_fs):
    """Legacy workspace/fileset/path URLs should be normalized to workspace/fileset#path."""
    mock_fileset_fs.info.return_value = {"name": "test", "size": 100, "type": "file"}

    fileset_manager.download_from_url(
        url="fileset://default/job-results-jobid-123/legacy/path/test.txt",
        local_dir=tmp_path,
    )

    mock_fileset_fs.info.assert_called_once_with("default/job-results-jobid-123#legacy/path/test.txt")
    mock_fileset_fs.get_file.assert_called_once_with(
        "default/job-results-jobid-123#legacy/path/test.txt",
        str(tmp_path / "test.txt"),
    )


def test_fileset_download_from_legacy_url_normalizes_to_hash_directory_ref(tmp_path, fileset_manager, mock_fileset_fs):
    """Legacy workspace/fileset/path directory URLs should still download correctly."""
    mock_fileset_fs.info.return_value = {"name": "test", "size": 0, "type": "directory"}

    fileset_manager.download_from_url(
        url="fileset://default/job-results-jobid-123/legacy/path/mydir",
        local_dir=tmp_path,
    )

    mock_fileset_fs.info.assert_called_once_with("default/job-results-jobid-123#legacy/path/mydir")
    mock_fileset_fs.get.assert_called_once_with(
        "default/job-results-jobid-123#legacy/path/mydir",
        str(tmp_path),
        recursive=True,
    )


def test_fileset_file_manager_multiple_sequential_operations(tmp_path, fileset_manager, mock_fileset_fs):
    """Test that FilesetFileManager can perform multiple sequential operations.

    This exercises the sync filesystem surface used by the sync file manager.
    """
    mock_fileset_fs.info.return_value = {"name": "test", "size": 0, "type": "file"}

    # Multiple operations in sequence should stay on the sync filesystem surface.
    fileset_manager.validate_storage()

    test_file = tmp_path / "test.txt"
    test_file.write_text("content")
    fileset_manager.upload(test_file, "remote/test.txt")

    fileset_manager.download_from_url("default/job-results-jobid-123#remote/file.txt", tmp_path)

    # Verify all operations were called
    assert mock_fileset_fs.info.call_count >= 1
    assert mock_fileset_fs.put_file.called
    assert mock_fileset_fs.get_file.called


async def test_fileset_upload_directory_with_ignore_patterns(tmp_path, mock_fileset_fs):
    """Test async uploading a directory with ignore_patterns filters files."""
    from nmp.common.entities import DEFAULT_WORKSPACE
    from nmp.common.jobs.file_manager import AsyncFilesetFileManager

    test_dir = tmp_path / "mydir"
    test_dir.mkdir()
    (test_dir / "file.txt").write_text("keep this")
    (test_dir / "ignore.pyc").write_text("skip this")
    subdir = test_dir / "subdir"
    subdir.mkdir()
    (subdir / "nested.txt").write_text("keep nested")
    (subdir / "nested.pyc").write_text("skip nested")

    async_manager = AsyncFilesetFileManager(
        workspace=DEFAULT_WORKSPACE,
        fileset_name="job-results-jobid-123",
        filesystem=mock_fileset_fs,
    )

    await async_manager.upload(test_dir, "remote/mydir", ignore_patterns="*.pyc")

    # Should upload only .txt files (2 calls)
    assert mock_fileset_fs._put_file.call_count == 2

    # Verify the correct files were uploaded
    call_args_list = mock_fileset_fs._put_file.call_args_list
    uploaded_files = [call[0][0] for call in call_args_list]
    assert any("file.txt" in f for f in uploaded_files)
    assert any("nested.txt" in f for f in uploaded_files)
    assert not any(".pyc" in f for f in uploaded_files)
