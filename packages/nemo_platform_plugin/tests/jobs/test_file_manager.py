# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path
from typing import cast

import pytest
from filesets import AsyncFilesetFileSystem, FilesetFileSystem
from nemo_platform_plugin.jobs.file_manager import AsyncFilesetFileManager, FilesetFileManager


class _FakeFilesetFileSystem:
    def __init__(self) -> None:
        self.downloads: list[tuple[str, str, bool]] = []

    def info(self, path: str) -> object:
        raise AssertionError(f"info should not be called for trailing-slash refs: {path}")

    def get(self, rpath: str, lpath: str, recursive: bool = True) -> None:
        self.downloads.append((rpath, lpath, recursive))
        local_path = Path(lpath)
        local_path.mkdir(parents=True, exist_ok=True)
        (local_path / "context.txt").write_text("downloaded")


class _FakeAsyncFilesetFileSystem:
    def __init__(self) -> None:
        self.downloads: list[tuple[str, str, bool]] = []

    async def _info(self, path: str) -> object:
        raise AssertionError(f"_info should not be called for trailing-slash refs: {path}")

    async def _get(self, rpath: str, lpath: str, recursive: bool = True) -> None:
        self.downloads.append((rpath, lpath, recursive))
        local_path = Path(lpath)
        local_path.mkdir(parents=True, exist_ok=True)
        (local_path / "context.txt").write_text("downloaded")


def test_fileset_manager_downloads_trailing_slash_ref_without_info(tmp_path: Path) -> None:
    fs = _FakeFilesetFileSystem()
    manager = FilesetFileManager(
        workspace="default",
        fileset_name="inputs",
        filesystem=cast(FilesetFileSystem, fs),
        ensure_fileset_exists=False,
    )

    result = manager.download_from_url("default/inputs#project/", local_dir=tmp_path)

    assert fs.downloads == [("default/inputs#project/", str(tmp_path), True)]
    assert result.path == tmp_path
    assert result.tmp_dir == tmp_path
    assert (tmp_path / "context.txt").read_text() == "downloaded"


@pytest.mark.asyncio
async def test_async_fileset_manager_downloads_trailing_slash_ref_without_info(tmp_path: Path) -> None:
    fs = _FakeAsyncFilesetFileSystem()
    manager = AsyncFilesetFileManager(
        workspace="default",
        fileset_name="inputs",
        filesystem=cast(AsyncFilesetFileSystem, fs),
        ensure_fileset_exists=False,
    )

    result = await manager.download_from_url("default/inputs#project/", local_dir=tmp_path)

    assert fs.downloads == [("default/inputs#project/", str(tmp_path), True)]
    assert result.path == tmp_path
    assert result.tmp_dir == tmp_path
    assert (tmp_path / "context.txt").read_text() == "downloaded"
