# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""`fetch` against a fake Files service that reports paths the way the real one does."""

from __future__ import annotations

import io
import tarfile
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Literal, cast

import pytest
from nemo_builder_plugin.run import fetch
from nemo_builder_plugin.run.fetch import ArchiveRefused, clear_earlier_attempts, fetch_source
from nemo_builder_plugin.run.sandbox import mounts
from nemo_builder_plugin.steps import ContextSource, SandboxGroup, SandboxImage, WorkLayout
from nemo_helix_plugin.files.client import FilesClient

JOB_SLICE = "jobs/default/abc"


@dataclass
class _Entry:
    path: str


@dataclass
class _Listing:
    data: list[_Entry]


class FakeFiles:
    """Entry paths are root-relative even in a narrowed listing, as in the real service."""

    def __init__(self, files: dict[str, bytes]) -> None:
        self.files = files

    def list_files(self, *, workspace: str, name: str, query_params: dict | None = None):
        prefix = (query_params or {}).get("path")
        entries = [_Entry(p) for p in self.files if prefix is None or p.startswith(f"{prefix.rstrip('/')}/")]

        class _Response:
            def data(self) -> _Listing:
                return _Listing(entries)

        return _Response()

    def download_file(self, *, workspace: str, name: str, path: str):
        content = self.files[path]

        class _Body:
            def read(self) -> bytes:
                return content

            @contextmanager
            def stream(self, chunk_size: int | None = None) -> Iterator[Iterator[bytes]]:
                size = chunk_size or 7
                yield iter([content[i : i + size] for i in range(0, len(content), size)])

        return _Body()


def _client(files: dict[str, bytes]) -> FilesClient:
    # A structural stand-in for the two calls `fetch` makes; the generated client is not a Protocol.
    return cast(FilesClient, FakeFiles(files))


FILESET = {"tests/Dockerfile": b"FROM tests", "env/Dockerfile": b"FROM env", "README": b"hi"}


def _sandbox_context(tmp_path: Path, source: ContextSource) -> Path:
    """The directory the sandbox would see, found the way `supervise` finds it: from the mount."""
    group = SandboxGroup(
        source=source, images=[SandboxImage(image="demo-1-0", platform="linux/amd64", dockerfile="Dockerfile")]
    )
    context_mount = mounts(group, JOB_SLICE)[0]
    return tmp_path / PurePosixPath(context_mount.sub_path).relative_to(JOB_SLICE)


class TestWhatTheSandboxSees:
    def test_a_subtree_lands_where_the_sandbox_mounts_it(self, tmp_path: Path) -> None:
        source = ContextSource(fileset="default/fs-a", context_path="tests")
        fetch_source(_client(FILESET), WorkLayout(PurePosixPath(tmp_path)), source, workspace="default")

        context = _sandbox_context(tmp_path, source)
        assert (context / "Dockerfile").read_bytes() == b"FROM tests"
        assert not (context / "tests").exists()
        assert not (tmp_path / "context/default/fs-a/env").exists()

    def test_a_whole_fileset_download_serves_its_subtrees(self, tmp_path: Path) -> None:
        fetch_source(
            _client(FILESET),
            WorkLayout(PurePosixPath(tmp_path)),
            ContextSource(fileset="default/fs-a"),
            workspace="default",
        )

        for subtree, content in (("tests", b"FROM tests"), ("env", b"FROM env")):
            context = _sandbox_context(tmp_path, ContextSource(fileset="default/fs-a", context_path=subtree))
            assert (context / "Dockerfile").read_bytes() == content

    def test_nothing_but_the_fileset_is_written_to_its_context(self, tmp_path: Path) -> None:
        layout = WorkLayout(PurePosixPath(tmp_path))
        count = fetch_source(_client(FILESET), layout, ContextSource(fileset="default/fs-a"), workspace="default")
        written = sorted(
            p.relative_to(tmp_path / "context/default/fs-a").as_posix() for p in tmp_path.rglob("*") if p.is_file()
        )
        assert count == len(FILESET) and written == sorted(FILESET)


class TestEachRunStartsClean:
    def test_an_earlier_runs_contexts_and_layouts_are_removed(self, tmp_path: Path) -> None:
        layout = WorkLayout(PurePosixPath(tmp_path))
        stale_context = Path(layout.context(ContextSource(fileset="default/fs-a"))) / "removed-since"
        stale_unpacked = Path(layout.archive("default/fs-a", "task.tar")) / "removed-since"
        stale_download = Path(layout.downloads) / "tmp-archive"
        stale_layout = Path(layout.output("demo-1-0")) / "index.json"
        for path in (stale_context, stale_unpacked, stale_download, stale_layout):
            path.parent.mkdir(parents=True)
            path.write_text("old")

        clear_earlier_attempts(layout)
        assert not any(p.exists() for p in (stale_context, stale_unpacked, stale_download, stale_layout))


TarMode = Literal["w", "w:gz", "w:xz"]


def _tar(entries: dict[str, bytes], *, mode: TarMode = "w:gz", modes: dict[str, int] | None = None) -> bytes:
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode=mode) as tar:
        for name, content in entries.items():
            info = tarfile.TarInfo(name)
            info.size, info.mode = len(content), (modes or {}).get(name, 0o644)
            tar.addfile(info, io.BytesIO(content))
    return buffer.getvalue()


def _symlink_tar(name: str, target: str) -> bytes:
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w") as tar:
        info = tarfile.TarInfo(name)
        info.type, info.linkname = tarfile.SYMTYPE, target
        tar.addfile(info)
    return buffer.getvalue()


TASK = {
    "environment/Dockerfile": b"FROM env",
    "environment/run.sh": b"#!/bin/sh",
    "tests/Dockerfile": b"FROM tests",
    "task.toml": b"",
}


def _fetch_archive(tmp_path: Path, archive: bytes, *, name: str = "tb/task.tar.gz") -> int:
    return fetch_source(
        _client({name: archive, "README": b"hi"}),
        WorkLayout(PurePosixPath(tmp_path)),
        ContextSource(fileset="default/fs-a", archive=name),
        workspace="default",
    )


class TestArchives:
    @pytest.mark.parametrize(("mode", "name"), [("w:gz", "tb/task.tar.gz"), ("w", "task.tar"), ("w:xz", "task.txz")])
    def test_each_directory_of_an_unpacked_archive_is_where_the_sandbox_mounts_it(
        self, tmp_path: Path, mode: TarMode, name: str
    ) -> None:
        assert _fetch_archive(tmp_path, _tar(TASK, mode=mode), name=name) == len(TASK)

        for directory, content in (("environment", b"FROM env"), ("tests", b"FROM tests")):
            source = ContextSource(fileset="default/fs-a", archive=name, context_path=directory)
            assert (_sandbox_context(tmp_path, source) / "Dockerfile").read_bytes() == content

    def test_only_the_archive_is_downloaded_and_nothing_of_it_is_kept(self, tmp_path: Path) -> None:
        _fetch_archive(tmp_path, _tar(TASK))
        assert not (tmp_path / "context").exists()
        assert not any((tmp_path / "downloads").iterdir())

    def test_an_executable_stays_executable(self, tmp_path: Path) -> None:
        _fetch_archive(tmp_path, _tar(TASK, modes={"environment/run.sh": 0o755}))
        source = ContextSource(fileset="default/fs-a", archive="tb/task.tar.gz", context_path="environment")
        assert (_sandbox_context(tmp_path, source) / "run.sh").stat().st_mode & 0o111

    @pytest.mark.parametrize(
        "archive", [_tar({"../escaped": b"!"}), _symlink_tar("link", "/etc/passwd")], ids=["climbing", "a-link-out"]
    )
    def test_an_entry_that_would_leave_the_archives_directory_is_refused(self, tmp_path: Path, archive: bytes) -> None:
        with pytest.raises(ArchiveRefused, match="default/fs-a/tb/task.tar.gz"):
            _fetch_archive(tmp_path, archive)
        assert not (tmp_path / "unpacked/default/fs-a/tb/escaped").exists()

    def test_an_absolute_entry_lands_inside_the_archives_directory(self, tmp_path: Path) -> None:
        _fetch_archive(tmp_path, _tar({"/etc/escaped": b"!"}))
        assert (tmp_path / "unpacked/default/fs-a/tb/task.tar.gz/etc/escaped").read_bytes() == b"!"

    def test_a_file_that_is_not_a_tar_is_refused(self, tmp_path: Path) -> None:
        with pytest.raises(ArchiveRefused, match="isn't a tar archive"):
            _fetch_archive(tmp_path, b"not a tar at all")

    def test_too_many_entries_are_refused_before_the_one_over(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(fetch, "MAX_UNPACKED_ENTRIES", 3)
        with pytest.raises(ArchiveRefused, match="more than 3 entries"):
            _fetch_archive(tmp_path, _tar(TASK))
        assert len([p for p in (tmp_path / "unpacked").rglob("*") if p.is_file()]) == 3

    def test_too_many_bytes_are_refused_before_they_are_written(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(fetch, "MAX_UNPACKED_BYTES", 10)
        with pytest.raises(ArchiveRefused, match="more than 10 bytes"):
            _fetch_archive(tmp_path, _tar({"a": b"12345678", "b": b"12345678"}))
        assert not (tmp_path / "unpacked/default/fs-a/tb/task.tar.gz/b").exists()


class TestHostileListings:
    def test_an_entry_path_that_escapes_the_fileset_is_refused(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="escapes"):
            fetch_source(
                _client({"../other-job/x": b"!"}),
                WorkLayout(PurePosixPath(tmp_path)),
                ContextSource(fileset="default/fs-a"),
                workspace="default",
            )
