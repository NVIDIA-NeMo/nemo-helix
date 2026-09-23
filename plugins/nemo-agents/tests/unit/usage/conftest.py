# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared fixtures for ``nemo agents usage`` tests."""

from __future__ import annotations

import shutil
from collections.abc import Callable, Iterator
from pathlib import Path

import pytest
from nemo_helix_plugin.files.types import FilesetFileOutput, ListFilesetFilesResponse

FIXTURES_DIR = Path(__file__).parent / "fixtures"


def _staged_files(root: Path) -> list[str]:
    """Relative leaf paths under *root*, mirroring ``list_files`` output."""
    return [p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file()]


@pytest.fixture
def fixtures_dir() -> Path:
    """Path to the directory holding hand-written ``result.json`` fixtures."""
    return FIXTURES_DIR


@pytest.fixture
def tmp_run_dir(tmp_path: Path, fixtures_dir: Path) -> Path:
    """A single ``<ts>-<task>/`` run directory containing a result.json.

    Lays out ``<tmp_path>/20260429T220000Z-workspace-basic-mcp/result.json``
    populated from the ok-with-tokens fixture.  Use the
    :func:`run_dir_with` helper to override the source fixture.
    """
    run = tmp_path / "20260429T220000Z-workspace-basic-mcp"
    run.mkdir()
    shutil.copy(fixtures_dir / "result-ok-with-tokens.json", run / "result.json")
    return run


@pytest.fixture
def tmp_natjobs_dir(tmp_path: Path, fixtures_dir: Path) -> Path:
    """A ``nat-jobs/`` parent containing four runs (one per fixture).

    Layout:

    .. code-block:: text

        <tmp_path>/nat-jobs/
            20260429T220000Z-workspace-basic-mcp/result.json   # ok-with-tokens
            20260429T230000Z-secrets-crud-cli/result.json      # ok-null-tokens
            20260429T230500Z-files-upload-mcp/result.json      # failed-agent
            20260429T231000Z-models-list-mcp/result.json       # error-build
    """
    natjobs = tmp_path / "nat-jobs"
    natjobs.mkdir()
    layout = [
        ("20260429T220000Z-workspace-basic-mcp", "result-ok-with-tokens.json"),
        ("20260429T230000Z-secrets-crud-cli", "result-ok-null-tokens.json"),
        ("20260429T230500Z-files-upload-mcp", "result-failed-agent.json"),
        ("20260429T231000Z-models-list-mcp", "result-error-build.json"),
    ]
    for run_name, fixture_name in layout:
        run = natjobs / run_name
        run.mkdir()
        shutil.copy(fixtures_dir / fixture_name, run / "result.json")
    return natjobs


class _FakeFilesResponse:
    """NemoResponse-like stand-in whose ``.data()`` returns the listing model."""

    def __init__(self, listing: ListFilesetFilesResponse) -> None:
        self._listing = listing

    def data(self) -> ListFilesetFilesResponse:
        return self._listing


class _FakeBinaryResponse:
    """NemoBinaryResponse-like stand-in for ``download_file``."""

    def __init__(self, content: bytes) -> None:
        self._content = content

    def read(self) -> bytes:
        return self._content


class FakeFiles:
    """Stand-in for the typed ``FilesClient`` download contract.

    Enumerates *staged_dir* via ``list_files`` and serves each file's bytes via
    ``download_file``, letting tests pre-stage a directory tree and assert it
    gets delivered to the expected destination under the same relative paths.
    """

    def __init__(self, staged_dir: Path) -> None:
        self._staged = staged_dir
        self.list_calls: list[dict[str, object]] = []
        self.calls: list[dict[str, object]] = []

    def list_files(
        self,
        *,
        workspace: str | None = None,
        name: str,
        query_params=None,
    ) -> _FakeFilesResponse:
        self.list_calls.append({"name": name, "workspace": workspace})
        entries = [
            FilesetFileOutput(
                path=rel,
                size=(self._staged / rel).stat().st_size,
                file_ref="",
                file_url="",
            )
            for rel in _staged_files(self._staged)
        ]
        return _FakeFilesResponse(ListFilesetFilesResponse(data=entries))

    def download_file(
        self,
        *,
        workspace: str | None = None,
        name: str,
        path: str,
    ) -> _FakeBinaryResponse:
        self.calls.append({"name": name, "workspace": workspace, "path": path})
        return _FakeBinaryResponse((self._staged / path).read_bytes())


class FakeSDK:
    """Stand-in for ``NeMoHelix``; ``build_files_client`` hands out FakeFiles."""

    def __init__(self, staged_dir: Path) -> None:
        self.files = FakeFiles(staged_dir)

    def build_files_client(self) -> FakeFiles:
        return self.files


@pytest.fixture
def fake_sdk_factory(monkeypatch) -> Iterator[Callable[[Path], FakeSDK]]:
    """Yield a factory for :class:`FakeSDK` bound to the typed ``FilesClient``.

    Each produced instance is wired as the ``client_from_platform`` result
    inside ``fileset`` so ``fileset_path`` consumes the fake client.
    """

    def make(staged_dir: Path) -> FakeSDK:
        sdk = FakeSDK(staged_dir)
        monkeypatch.setattr(
            "nemo_agents_plugin.usage.sources.fileset.client_from_platform",
            lambda _platform, _client_cls: sdk.build_files_client(),
        )
        return sdk

    yield make
