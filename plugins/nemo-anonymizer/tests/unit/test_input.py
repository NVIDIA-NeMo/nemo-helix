# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import threading
from pathlib import Path

import pytest
from nemo_anonymizer_plugin.app import input as input_module
from nemo_anonymizer_plugin.app.errors import AnonymizerInvalidConfigError
from nemo_anonymizer_plugin.app.input import (
    AnonymizerInputSpec,
    classify_input_source,
    prepare_anonymizer_input,
    prepare_anonymizer_input_async,
    validate_anonymizer_input_source,
)
from nemo_platform import AsyncNeMoPlatform, NeMoPlatform
from nemo_platform_plugin.files.client import AsyncFilesClient, FilesClient
from nemo_platform_plugin.jobs.file_manager import TmpDirPath

FilesetClient = FilesClient | AsyncFilesClient


class FakeFilesetFileSystem:
    def __init__(self, *, client: FilesetClient) -> None:
        self.client = client


def _async_platform() -> AsyncNeMoPlatform:
    return AsyncNeMoPlatform(base_url="http://platform.test", workspace="team-a")


def _platform() -> NeMoPlatform:
    return NeMoPlatform(base_url="http://platform.test", workspace="team-a")


def _patch_files_client(monkeypatch: pytest.MonkeyPatch) -> None:
    def client_from_platform(
        sdk: NeMoPlatform | AsyncNeMoPlatform,
        client_cls: type[FilesClient] | type[AsyncFilesClient],
    ) -> FilesetClient:
        base_url = str(sdk.base_url).rstrip("/")
        if client_cls is FilesClient:
            return FilesClient(base_url=base_url, workspace=sdk.workspace)
        return AsyncFilesClient(base_url=base_url, workspace=sdk.workspace)

    monkeypatch.setattr(input_module, "FilesetFileSystem", FakeFilesetFileSystem)
    monkeypatch.setattr(input_module, "AsyncFilesetFileSystem", FakeFilesetFileSystem)
    monkeypatch.setattr(input_module, "client_from_platform", client_from_platform)


def test_classifies_supported_sources() -> None:
    assert classify_input_source("https://example.com/input.csv") == "http"
    assert classify_input_source("fileset://team-a/fs#input.csv") == "fileset"
    assert classify_input_source("team-a/fs#input.csv") == "fileset"
    assert classify_input_source("/tmp/input.csv") == "local"
    assert classify_input_source("/data/raw#2026.csv") == "local"
    assert classify_input_source("./raw#2026.csv") == "local"


def test_existing_relative_hash_path_wins_over_fileset_shorthand(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    csv = tmp_path / "fs#input.csv"
    csv.write_text("text\nhello\n")
    monkeypatch.chdir(tmp_path)

    assert classify_input_source("fs#input.csv") == "local"


def test_remote_validation_rejects_local_path() -> None:
    with pytest.raises(AnonymizerInvalidConfigError, match="local path"):
        validate_anonymizer_input_source(
            AnonymizerInputSpec(source="/tmp/input.csv"),
            workspace="team-a",
            allow_local_paths=False,
        )


def test_fileset_validation_requires_file_fragment() -> None:
    with pytest.raises(AnonymizerInvalidConfigError, match="#path"):
        validate_anonymizer_input_source(
            AnonymizerInputSpec(source="fileset://team-a/fs"),
            workspace="team-a",
            allow_local_paths=False,
        )


@pytest.mark.asyncio
async def test_prepare_fileset_materializes_before_building_upstream_input(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    downloaded = tmp_path / "input.csv"
    downloaded.write_text("text\nhello\n")
    tmp_dir_path = TmpDirPath(path=downloaded, tmp_dir=tmp_path)
    captured: dict[str, object] = {}

    class FakeAsyncFilesetFileManager:
        def __init__(
            self,
            *,
            workspace: str,
            fileset_name: str,
            filesystem: FakeFilesetFileSystem,
            ensure_fileset_exists: bool,
        ) -> None:
            captured["workspace"] = workspace
            captured["fileset_name"] = fileset_name
            captured["filesystem"] = filesystem
            captured["ensure_fileset_exists"] = ensure_fileset_exists

        async def download_from_url(self, url: str) -> TmpDirPath:
            captured["url"] = url
            return tmp_dir_path

    monkeypatch.setattr(input_module, "AsyncFilesetFileManager", FakeAsyncFilesetFileManager)
    _patch_files_client(monkeypatch)

    prepared = await prepare_anonymizer_input_async(
        AnonymizerInputSpec(source="fs#data/input.csv", text_column="body"),
        sdk=_async_platform(),
        workspace="team-a",
        allow_local_paths=False,
    )

    assert captured["workspace"] == "team-a"
    assert captured["fileset_name"] == "fs"
    assert captured["url"] == "team-a/fs#data/input.csv"
    assert captured["ensure_fileset_exists"] is False
    assert prepared.input.source == str(downloaded)
    assert prepared.input.text_column == "body"


@pytest.mark.asyncio
async def test_prepare_fileset_rejects_downloaded_directory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    download_dir = tmp_path / "download"
    download_dir.mkdir()
    tmp_dir_path = TmpDirPath(path=download_dir, tmp_dir=download_dir)

    class FakeAsyncFilesetFileManager:
        def __init__(
            self,
            *,
            workspace: str,
            fileset_name: str,
            filesystem: FakeFilesetFileSystem,
            ensure_fileset_exists: bool,
        ) -> None:
            del workspace, fileset_name, filesystem, ensure_fileset_exists

        async def download_from_url(self, url: str) -> TmpDirPath:
            return tmp_dir_path

    monkeypatch.setattr(input_module, "AsyncFilesetFileManager", FakeAsyncFilesetFileManager)
    _patch_files_client(monkeypatch)

    with pytest.raises(AnonymizerInvalidConfigError, match="directory"):
        await prepare_anonymizer_input_async(
            AnonymizerInputSpec(source="fs#data"),
            sdk=_async_platform(),
            workspace="team-a",
            allow_local_paths=False,
        )
    assert not download_dir.exists()


@pytest.mark.asyncio
async def test_prepare_fileset_cleans_download_when_upstream_input_rejects_data(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    download_dir = tmp_path / "download"
    download_dir.mkdir()
    downloaded = download_dir / "input.csv"
    downloaded.write_text("text\nhello\n")
    tmp_dir_path = TmpDirPath(path=downloaded, tmp_dir=download_dir)

    class FakeAsyncFilesetFileManager:
        def __init__(
            self,
            *,
            workspace: str,
            fileset_name: str,
            filesystem: FakeFilesetFileSystem,
            ensure_fileset_exists: bool,
        ) -> None:
            del workspace, fileset_name, filesystem, ensure_fileset_exists

        async def download_from_url(self, url: str) -> TmpDirPath:
            return tmp_dir_path

    monkeypatch.setattr(input_module, "AsyncFilesetFileManager", FakeAsyncFilesetFileManager)
    _patch_files_client(monkeypatch)

    def raise_bad_input(data: AnonymizerInputSpec, *, source: str) -> input_module.AnonymizerInput:
        del data, source
        raise AnonymizerInvalidConfigError("bad input")

    monkeypatch.setattr(input_module, "_make_upstream_input", raise_bad_input)

    with pytest.raises(AnonymizerInvalidConfigError, match="bad input"):
        await prepare_anonymizer_input_async(
            AnonymizerInputSpec(source="fs#input.csv"),
            sdk=_async_platform(),
            workspace="team-a",
            allow_local_paths=False,
        )
    assert not download_dir.exists()


def test_prepare_fileset_sync_cleans_download_when_upstream_input_rejects_data(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    download_dir = tmp_path / "download"
    download_dir.mkdir()
    downloaded = download_dir / "input.csv"
    downloaded.write_text("text\nhello\n")
    tmp_dir_path = TmpDirPath(path=downloaded, tmp_dir=download_dir)

    class FakeFilesetFileManager:
        def __init__(
            self,
            *,
            workspace: str,
            fileset_name: str,
            filesystem: FakeFilesetFileSystem,
            ensure_fileset_exists: bool,
        ) -> None:
            del workspace, fileset_name, filesystem, ensure_fileset_exists

        def download_from_url(self, url: str) -> TmpDirPath:
            return tmp_dir_path

    monkeypatch.setattr(input_module, "FilesetFileManager", FakeFilesetFileManager)
    _patch_files_client(monkeypatch)

    def raise_bad_input(data: AnonymizerInputSpec, *, source: str) -> input_module.AnonymizerInput:
        del data, source
        raise AnonymizerInvalidConfigError("bad input")

    monkeypatch.setattr(input_module, "_make_upstream_input", raise_bad_input)

    with pytest.raises(AnonymizerInvalidConfigError, match="bad input"):
        prepare_anonymizer_input(
            AnonymizerInputSpec(source="fs#input.csv"),
            sdk=_platform(),
            workspace="team-a",
            allow_local_paths=False,
        )
    assert not download_dir.exists()


@pytest.mark.asyncio
async def test_prepare_fileset_with_sync_sdk_runs_in_worker_thread(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    event_loop_thread = threading.get_ident()
    sentinel_path = tmp_path / "input.csv"
    sentinel_path.write_text("text\nhello\n")
    sentinel = input_module.AnonymizerInput(source=str(sentinel_path))
    captured: dict[str, object] = {}

    def fake_prepare_anonymizer_input(
        data: AnonymizerInputSpec,
        *,
        sdk: NeMoPlatform,
        workspace: str,
        allow_local_paths: bool,
    ) -> input_module.PreparedAnonymizerInput:
        captured["thread"] = threading.get_ident()
        captured["data"] = data
        captured["sdk"] = sdk
        captured["workspace"] = workspace
        captured["allow_local_paths"] = allow_local_paths
        return input_module.PreparedAnonymizerInput(input=sentinel)

    sdk = _platform()
    monkeypatch.setattr(input_module, "prepare_anonymizer_input", fake_prepare_anonymizer_input)

    prepared = await prepare_anonymizer_input_async(
        AnonymizerInputSpec(source="fs#input.csv"),
        sdk=sdk,
        workspace="team-a",
        allow_local_paths=False,
    )

    assert prepared.input is sentinel
    assert captured["thread"] != event_loop_thread
    assert captured["sdk"] is sdk
    assert captured["workspace"] == "team-a"
    assert captured["allow_local_paths"] is False
