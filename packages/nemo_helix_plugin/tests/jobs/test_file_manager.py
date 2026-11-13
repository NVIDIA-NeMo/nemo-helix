# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import cast
from unittest.mock import AsyncMock, Mock

import pytest
from filesets import AsyncFilesetFileSystem, FilesetFileSystem
from nemo_helix_plugin.jobs.file_manager import AsyncFilesetFileManager, FilesetFileManager
from pytest_mock import MockerFixture


@pytest.mark.asyncio
@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("failure_count", [0, 1, 2])
async def test_directory_upload_attempts_remaining_files(
    tmp_path: Path, mocker: MockerFixture, asynchronous: bool, failure_count: int
) -> None:
    """Retain later nested files and report original errors after all nonignored files are attempted."""
    paths = ["first.txt", "nested/second.txt", "nested/deeper/last.txt", "ignored.tmp"]
    for path in paths:
        local = tmp_path / path
        local.parent.mkdir(parents=True, exist_ok=True)
        local.write_text(path)
    # Fix traversal order so failures demonstrably precede successful nested uploads.
    listing = "_list_local_files" if asynchronous else "_list_local_files_sync"
    mocker.patch(f"nemo_helix_plugin.jobs.file_manager.{listing}", return_value=paths)
    errors = [OSError(f"upload {index} failed") for index in range(failure_count)]
    effects = [*errors, *([None] * (3 - failure_count))]
    if asynchronous:
        fs = Mock(spec=AsyncFilesetFileSystem)
        upload = fs._put_file = AsyncMock(side_effect=effects)
        manager = AsyncFilesetFileManager("default", "results", fs, ensure_fileset_exists=False)
    else:
        fs = Mock(spec=FilesetFileSystem)
        upload = fs.put_file = Mock(side_effect=effects)
        manager = FilesetFileManager("default", "results", fs, ensure_fileset_exists=False)

    async def run_upload():
        """Exercise the public upload entry point for either manager."""
        if isinstance(manager, AsyncFilesetFileManager):
            return await manager.upload(tmp_path, "bundle", ignore_patterns="*.tmp")
        return manager.upload(tmp_path, "bundle", ignore_patterns="*.tmp")

    if errors:
        with pytest.raises(OSError if failure_count == 1 else ExceptionGroup) as raised:
            await run_upload()
        if failure_count == 1:
            assert raised.value is errors[0]
        else:
            assert isinstance(raised.value, ExceptionGroup)
            assert raised.value.exceptions == tuple(errors)
        for index, error in enumerate(errors):
            assert paths[index] in error.__notes__[0]
    else:
        assert await run_upload() == manager.url("bundle")

    assert [call.args for call in upload.call_args_list] == [
        (str(tmp_path / path), f"default/results#bundle/{path}") for path in paths[:3]
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("interruption", [KeyboardInterrupt, asyncio.CancelledError])
async def test_directory_upload_stops_immediately_on_interruption(
    tmp_path: Path, mocker: MockerFixture, asynchronous: bool, interruption: type[BaseException]
) -> None:
    """An interruption stops uploads even after an earlier ordinary upload failure."""
    listing = "_list_local_files" if asynchronous else "_list_local_files_sync"
    mocker.patch(f"nemo_helix_plugin.jobs.file_manager.{listing}", return_value=["first", "second", "third"])
    stop = interruption()
    if asynchronous:
        fs = Mock(spec=AsyncFilesetFileSystem)
        upload = fs._put_file = AsyncMock(side_effect=[OSError("failed"), stop, None])
        manager = AsyncFilesetFileManager("default", "results", fs, ensure_fileset_exists=False)
    else:
        fs = Mock(spec=FilesetFileSystem)
        upload = fs.put_file = Mock(side_effect=[OSError("failed"), stop, None])
        manager = FilesetFileManager("default", "results", fs, ensure_fileset_exists=False)

    with pytest.raises(interruption) as raised:
        if isinstance(manager, AsyncFilesetFileManager):
            await manager.upload(tmp_path, "bundle")
        else:
            manager.upload(tmp_path, "bundle")
    assert raised.value is stop
    assert upload.call_count == 2


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


class _RecordingUploadFileSystem:
    def __init__(self) -> None:
        self.uploads: dict[str, str] = {}

    def put_file(self, lpath: str, rpath: str) -> None:
        self.uploads[rpath] = Path(lpath).read_text()

    async def _put_file(self, lpath: str, rpath: str) -> None:
        self.put_file(lpath, rpath)


@pytest.mark.asyncio
async def test_directory_upload_keeps_regular_files_and_skips_symlinks(tmp_path: Path) -> None:
    outside = tmp_path / "worker"
    (outside / "secrets").mkdir(parents=True)
    (outside / "token").write_text("worker-token")
    (outside / "secrets" / "key").write_text("worker-key")
    bundle = tmp_path / "bundle"
    (bundle / "outputs" / ".hidden").mkdir(parents=True)
    (bundle / "run.json").write_text("run")
    (bundle / "outputs" / ".hidden" / "video.bin").write_text("video")
    (bundle / "outputs" / "token").symlink_to(outside / "token")
    (bundle / "outputs" / "secrets").symlink_to(outside / "secrets", target_is_directory=True)
    expected = {"out/run.json": "run", "out/outputs/.hidden/video.bin": "video"}

    sync_fs = _RecordingUploadFileSystem()
    FilesetFileManager(workspace="default", fileset_name="results", filesystem=cast(FilesetFileSystem, sync_fs)).upload(
        bundle, "out"
    )
    async_fs = _RecordingUploadFileSystem()
    await AsyncFilesetFileManager(
        workspace="default", fileset_name="results", filesystem=cast(AsyncFilesetFileSystem, async_fs)
    ).upload(bundle, "out")

    for fs in (sync_fs, async_fs):
        assert {rpath.split("#", 1)[1]: content for rpath, content in fs.uploads.items()} == expected
