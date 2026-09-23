# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import Mock, patch

import pytest
from data_designer_nemo.fileset_file_seed_reader import FilesetFileSeedReader, workspace_cvar
from data_designer_nemo.fileset_file_seed_source import FilesetFileSeedSource


def test_dataset_uri_with_workspace() -> None:
    path = "my-workspace/my-fileset#path/to/data.parquet"
    source = FilesetFileSeedSource(path=path)

    reader = FilesetFileSeedReader(Mock())
    reader.attach(source, Mock())

    assert reader.get_dataset_uri() == f"fileset://{path}"


def test_dataset_uri_no_workspace() -> None:
    request_workspace = "request-workspace"
    workspace_cvar.set(request_workspace)

    path = "my-fileset#path/to/data.parquet"
    source = FilesetFileSeedSource(path=path)

    reader = FilesetFileSeedReader(Mock())
    reader.attach(source, Mock())

    assert reader.get_dataset_uri() == f"fileset://{request_workspace}/{path}"


def test_create_duckdb_connection_requires_injected_sdk() -> None:
    with pytest.raises(RuntimeError, match="requires an injected NeMo Helix SDK"):
        FilesetFileSeedReader().create_duckdb_connection()


def test_create_duckdb_connection_uses_injected_sdk() -> None:
    sdk = Mock()
    conn = Mock()
    filesystem = Mock()

    with (
        patch("data_designer_nemo.fileset_file_seed_reader.duckdb.connect", return_value=conn),
        patch("data_designer_nemo.fileset_file_seed_reader.make_filesystem", return_value=filesystem),
    ):
        assert FilesetFileSeedReader(sdk).create_duckdb_connection() is conn

    conn.register_filesystem.assert_called_once_with(filesystem)
