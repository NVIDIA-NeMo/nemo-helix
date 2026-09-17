# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from unittest.mock import AsyncMock, Mock, patch

import data_designer.config as dd
import pytest
from data_designer.engine.resources.seed_reader import DirectorySeedReader, FileContentsSeedReader
from data_designer_nemo.context import DataDesignerExecutionContext
from data_designer_nemo.errors import NDDInvalidConfigError
from data_designer_nemo.fileset_file_seed_source import FilesetFileSeedSource
from data_designer_nemo.seed import _validate_seed_from_files_service, validate_seed
from nemo_platform import AsyncNeMoPlatform, NeMoPlatform
from nemo_platform_plugin.files.types import ListFilesQueryParams


def _list_files_response(paths: list[str]) -> Mock:
    return Mock(data=Mock(return_value=Mock(data=[Mock(path=path) for path in paths])))


def test_remote_context_includes_filesystem_seed_readers() -> None:
    readers = DataDesignerExecutionContext(Mock(spec=NeMoPlatform), "default").get_seed_readers()

    assert any(isinstance(reader, DirectorySeedReader) for reader in readers)
    assert any(isinstance(reader, FileContentsSeedReader) for reader in readers)


@pytest.mark.asyncio
async def test_validate_seed_returns_canonical_validated_filesystem_root() -> None:
    sdk = AsyncMock(spec=AsyncNeMoPlatform)
    files = Mock()
    files.get_fileset = AsyncMock()
    files.list_files = AsyncMock(return_value=_list_files_response(["corpus/a.md"]))

    builder = dd.DataDesignerConfigBuilder()
    builder.with_seed_dataset(dd.FileContentsSeedSource(path="docs#corpus", file_pattern="*.md"))
    config = builder.build()

    with patch("data_designer_nemo.seed.client_from_platform", return_value=files):
        validated_root = await validate_seed(config, "default", sdk)

    assert validated_root == "default/docs#corpus"
    files.get_fileset.assert_awaited_once_with(name="docs", workspace="default")
    files.list_files.assert_awaited_once_with(
        workspace="default",
        name="docs",
        query_params=ListFilesQueryParams(path="corpus"),
    )


@pytest.mark.asyncio
async def test_validate_seed_rejects_fileset_root_with_no_files() -> None:
    sdk = AsyncMock(spec=AsyncNeMoPlatform)
    files = Mock()
    files.get_fileset = AsyncMock()
    files.list_files = AsyncMock(return_value=_list_files_response([]))

    builder = dd.DataDesignerConfigBuilder()
    builder.with_seed_dataset(dd.DirectorySeedSource(path="docs"))
    config = builder.build()

    with pytest.raises(NDDInvalidConfigError, match="contains no files to use as seed data"):
        with patch("data_designer_nemo.seed.client_from_platform", return_value=files):
            await validate_seed(config, "default", sdk)

    files.list_files.assert_awaited_once_with(
        workspace="default",
        name="docs",
        query_params=None,
    )


@pytest.mark.asyncio
async def test_validate_seed_rejects_path_with_no_files() -> None:
    sdk = AsyncMock(spec=AsyncNeMoPlatform)
    files = Mock()
    files.get_fileset = AsyncMock()
    files.list_files = AsyncMock(return_value=_list_files_response([]))

    builder = dd.DataDesignerConfigBuilder()
    builder.with_seed_dataset(dd.FileContentsSeedSource(path="docs#corpus", file_pattern="*.md"))
    config = builder.build()

    with pytest.raises(NDDInvalidConfigError, match="contains no files to use as seed data"):
        with patch("data_designer_nemo.seed.client_from_platform", return_value=files):
            await validate_seed(config, "default", sdk)


@pytest.mark.asyncio
async def test_validate_seed_reports_missing_fileset_file() -> None:
    # FilesetFileSeedSource points at a single file, so the error should say "File ... not found"
    # rather than the directory-style "contains no files" message.
    sdk = AsyncMock(spec=AsyncNeMoPlatform)
    files = Mock()
    files.get_fileset = AsyncMock()
    files.list_files = AsyncMock(return_value=_list_files_response([]))

    seed_source = FilesetFileSeedSource(path="docs#corpus/missing.parquet")

    with pytest.raises(NDDInvalidConfigError, match=r"File 'corpus/missing.parquet' not found"):
        with patch("data_designer_nemo.seed.client_from_platform", return_value=files):
            await _validate_seed_from_files_service(seed_source, "default", sdk)
