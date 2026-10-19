# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import Mock, patch

import pytest
from data_designer.engine.resources.seed_reader import SeedReaderConfigError
from data_designer_nemo.fileset_filesystem_provider import FilesetFileSystemProvider


def test_create_context_roots_reader_in_canonical_fileset_ref() -> None:
    client = Mock()
    filesystem = Mock()

    with patch("data_designer_nemo.fileset_filesystem_provider.make_filesystem", return_value=filesystem) as make_fs:
        filesystem.async_impl = True
        filesystem.asynchronous = False
        context = FilesetFileSystemProvider(client, workspace="default").create_context(runtime_path="docs#corpus")

    make_fs.assert_called_once_with(client)
    assert str(context.root_path) == "default/docs#corpus"
    assert str(context.root_path / "data.parquet") == "default/docs#corpus/data.parquet"


def test_create_context_uses_canonical_source_paths_for_fileset_root() -> None:
    client = Mock()
    filesystem = Mock()

    with patch("data_designer_nemo.fileset_filesystem_provider.make_filesystem", return_value=filesystem):
        filesystem.async_impl = True
        filesystem.asynchronous = False
        context = FilesetFileSystemProvider(client, workspace="default").create_context(runtime_path="docs")

    assert str(context.root_path) == "default/docs"
    assert str(context.root_path / "data.parquet") == "default/docs#data.parquet"


def test_ensure_root_exists_skips_validated_roots() -> None:
    client = Mock()

    with patch("data_designer_nemo.fileset_filesystem_provider.make_filesystem") as make_fs:
        FilesetFileSystemProvider(
            client,
            workspace="default",
            validated_roots={"default/docs#corpus"},
        ).ensure_root_exists(runtime_path="docs#corpus")

    make_fs.assert_not_called()


def test_ensure_root_exists_reports_missing_fileset_path() -> None:
    client = Mock()
    filesystem = Mock()

    with patch("data_designer_nemo.fileset_filesystem_provider.make_filesystem", return_value=filesystem):
        filesystem.exists.side_effect = [False, True]
        provider = FilesetFileSystemProvider(client, workspace="default")

        with pytest.raises(SeedReaderConfigError, match="Path 'corpus' not found in fileset 'default/docs'"):
            provider.ensure_root_exists(runtime_path="docs#corpus")

    assert filesystem.exists.call_count == 2
