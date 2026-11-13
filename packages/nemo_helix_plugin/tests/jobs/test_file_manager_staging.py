# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Directory upload acquisition uses real files, including a disappearing clone lock."""

import asyncio
import errno
import logging
import os
import shutil
import tempfile
from collections.abc import Callable, Iterator
from functools import partial
from pathlib import Path
from threading import Event
from types import SimpleNamespace
from typing import NoReturn, cast
from unittest.mock import AsyncMock, Mock

import pytest
from filesets import AsyncFilesetFileSystem, FilesetFileSystem
from nemo_helix_plugin.files.client import AsyncFilesClient, FilesClient
from nemo_helix_plugin.jobs import file_manager
from nemo_helix_plugin.jobs.client import AsyncJobsClient, JobsClient
from nemo_helix_plugin.jobs.result_manager import AsyncResultManager, ResultManager


class _RecordingFileSystem:
    def __init__(self) -> None:
        self.uploads: dict[str, bytes] = {}
        self.local_paths: list[Path] = []

    def put_file(self, local: str, remote: str) -> None:
        self.local_paths.append(Path(local))
        self.uploads[remote] = Path(local).read_bytes()

    async def _put_file(self, local: str, remote: str) -> None:
        self.put_file(local, remote)

    def info(self, path: str) -> dict[str, str]:
        return {"type": "directory"}

    async def _info(self, path: str) -> dict[str, str]:
        return self.info(path)


@pytest.fixture
def upload_temporary_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    root = tmp_path / "upload-copies"
    root.mkdir()
    monkeypatch.setattr(
        file_manager,
        "tempfile",
        SimpleNamespace(TemporaryDirectory=partial(tempfile.TemporaryDirectory, dir=root)),
    )
    return root


def _manager(
    asynchronous: bool, fs: _RecordingFileSystem
) -> file_manager.FilesetFileManager | file_manager.AsyncFilesetFileManager:
    if asynchronous:
        return file_manager.AsyncFilesetFileManager("default", "results", cast(AsyncFilesetFileSystem, fs))
    return file_manager.FilesetFileManager("default", "results", cast(FilesetFileSystem, fs))


async def _upload(
    manager: file_manager.FilesetFileManager | file_manager.AsyncFilesetFileManager,
    source: Path,
    ignore_patterns: list[str] | str | None = None,
    *,
    stage: bool = True,
) -> str:
    if isinstance(manager, file_manager.AsyncFilesetFileManager):
        return await manager.upload(source, "bundle", ignore_patterns, stage=stage)
    return manager.upload(source, "bundle", ignore_patterns, stage=stage)


def test_staged_directory_survives_source_cleanup_and_keeps_hidden_user_files(tmp_path: Path) -> None:
    source = tmp_path / "result"
    required = {
        "summary.json": b'{"score_count":1,"error_count":0}',
        "trials.jsonl": b'{"status":"completed"}\n',
        "workspace/.git/user.lock": b"user artifact, not a runtime exclusion",
        "artifacts/trace.json": b'{"steps":[]}',
    }
    for relative, content in required.items():
        path = source / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
    staging = tmp_path / "staging"
    staging.mkdir()

    snapshot, files = file_manager._stage_local_files(source, staging, None)
    for relative in required:
        (source / relative).unlink()

    assert set(files) == set(required)
    assert {relative: (snapshot / relative).read_bytes() for relative in files} == required


def test_disappearing_clone_lock_restarts_complete_acquisition(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    source = tmp_path / "result"
    source.mkdir()
    (source / "summary.json").write_text("required summary")
    lock = source / "artifacts/.fabric/codex/custom-provider-home/.tmp/plugins-clone-test/.git/shallow.lock"
    lock.parent.mkdir(parents=True)
    lock.write_text("transient clone state")
    staging = tmp_path / "staging"
    staging.mkdir()
    list_files = file_manager._list_selected_files
    listings = 0

    def list_then_finish_clone(path: Path, ignore_patterns: list[str] | str | None = None) -> list[str]:
        nonlocal listings
        files = list_files(path, ignore_patterns=ignore_patterns)
        listings += 1
        if listings == 1:
            lock.unlink()
        return files

    # Schedule a real unlink at the traversal/open boundary, without replacing
    # pathlib, shutil, open, or any product service/client.
    monkeypatch.setattr(file_manager, "_list_selected_files", list_then_finish_clone)
    snapshot, files = file_manager._stage_local_files(source, staging, None)

    assert listings == 2
    assert files == ["summary.json"]
    assert (snapshot / "summary.json").read_text() == "required summary"
    assert not (snapshot / lock.relative_to(source)).exists()


def test_persistently_disappearing_required_file_fails_instead_of_being_omitted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "result"
    source.mkdir()
    required = source / "trials.jsonl"
    staging = tmp_path / "staging"
    staging.mkdir()
    list_files = file_manager._list_selected_files
    listings = 0

    def list_then_remove_required(path: Path, ignore_patterns: list[str] | str | None = None) -> list[str]:
        nonlocal listings
        required.write_text("required trial")
        files = list_files(path, ignore_patterns=ignore_patterns)
        listings += 1
        required.unlink()
        return files

    monkeypatch.setattr(file_manager, "_list_selected_files", list_then_remove_required)
    with pytest.raises(FileNotFoundError, match="trials.jsonl"):
        file_manager._stage_local_files(source, staging, None)

    assert listings == 3
    assert list(staging.iterdir()) == []


def test_missing_result_directory_is_not_published_as_empty(tmp_path: Path) -> None:
    staging = tmp_path / "staging"
    staging.mkdir()
    with pytest.raises(FileNotFoundError, match="Result directory no longer exists"):
        file_manager._stage_local_files(tmp_path / "missing", staging, None)


def test_staging_preserves_explicit_ignore_patterns(tmp_path: Path) -> None:
    source = tmp_path / "result"
    source.mkdir()
    (source / "summary.json").write_text("required")
    (source / "debug.log").write_text("explicitly ignored")
    staging = tmp_path / "staging"
    staging.mkdir()

    snapshot, files = file_manager._stage_local_files(source, staging, "*.log")

    assert files == ["summary.json"]
    assert (snapshot / "summary.json").read_text() == "required"
    assert not (snapshot / "debug.log").exists()


@pytest.mark.skipif(os.geteuid() == 0, reason="Root bypasses ordinary file read permissions")
def test_unreadable_required_file_remains_an_error(tmp_path: Path) -> None:
    source = tmp_path / "result"
    source.mkdir()
    required = source / "trials.jsonl"
    required.write_text("required")
    staging = tmp_path / "staging"
    staging.mkdir()
    required.chmod(0)
    try:
        with pytest.raises(PermissionError):
            file_manager._stage_local_files(source, staging, None)
    finally:
        required.chmod(0o600)


@pytest.mark.parametrize("asynchronous", [False, True], ids=["sync", "async"])
async def test_directory_upload_retries_disappearing_lock_and_cleans_staging(
    tmp_path: Path, upload_temporary_root: Path, monkeypatch: pytest.MonkeyPatch, asynchronous: bool
) -> None:
    source = tmp_path / "source"
    files = {"summary.json": b"summary", "workspace/.git/user.lock": b"user evidence", "ignored.log": b"omit"}
    for relative, data in files.items():
        path = source / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    lock = source / "artifacts/.fabric/codex/custom-provider-home/.tmp/clone/.git/shallow.lock"
    lock.parent.mkdir(parents=True)
    lock.write_bytes(b"runtime lock")
    list_files = file_manager._list_selected_files
    listings = 0

    def finish_clone_after_listing(path: Path, ignore_patterns: list[str] | str | None = None) -> list[str]:
        nonlocal listings
        listed = list_files(path, ignore_patterns=ignore_patterns)
        listings += 1
        if listings == 1:
            lock.unlink()
        return listed

    monkeypatch.setattr(file_manager, "_list_selected_files", finish_clone_after_listing)
    fs = _RecordingFileSystem()
    await _upload(_manager(asynchronous, fs), source, "*.log")

    assert listings == 2
    assert fs.uploads == {
        "default/results#bundle/summary.json": b"summary",
        "default/results#bundle/workspace/.git/user.lock": b"user evidence",
    }
    assert all(path.is_relative_to(upload_temporary_root) for path in fs.local_paths)
    assert not list(upload_temporary_root.iterdir())


@pytest.mark.parametrize("asynchronous", [False, True], ids=["sync", "async"])
async def test_directory_upload_reads_staged_bytes_after_source_changes(
    tmp_path: Path, upload_temporary_root: Path, monkeypatch: pytest.MonkeyPatch, asynchronous: bool
) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (source / "first").write_bytes(b"first original")
    (source / "second").write_bytes(b"second original")
    fs = _RecordingFileSystem()
    put_file = fs.put_file

    def change_then_upload(local: str, remote: str) -> None:
        if source.exists():
            (source / "second").write_bytes(b"changed")
            shutil.rmtree(source)
        put_file(local, remote)

    monkeypatch.setattr(fs, "put_file", change_then_upload)
    await _upload(_manager(asynchronous, fs), source)

    assert fs.uploads == {
        "default/results#bundle/first": b"first original",
        "default/results#bundle/second": b"second original",
    }
    assert not list(upload_temporary_root.iterdir())


@pytest.mark.parametrize("asynchronous", [False, True], ids=["sync", "async"])
async def test_source_root_removed_during_listing_cannot_publish_an_empty_result(
    tmp_path: Path, upload_temporary_root: Path, monkeypatch: pytest.MonkeyPatch, asynchronous: bool
) -> None:
    source = tmp_path / "source"
    source.mkdir()

    def remove_during_listing(path: Path, ignore_patterns: list[str] | str | None = None) -> list[str]:
        path.rmdir()
        return []

    monkeypatch.setattr(file_manager, "_list_selected_files", remove_during_listing)
    fs = _RecordingFileSystem()
    with pytest.raises(FileNotFoundError):
        await _upload(_manager(asynchronous, fs), source)
    assert not fs.local_paths
    assert not list(upload_temporary_root.iterdir())


@pytest.mark.parametrize("asynchronous", [False, True], ids=["sync", "async"])
@pytest.mark.skipif(os.geteuid() == 0, reason="Root bypasses ordinary directory permissions")
@pytest.mark.parametrize(
    ("directory", "ignore_patterns"), [("nested", None), ("cache", "cache"), ("cache.tmp", "*.tmp")]
)
async def test_unreadable_subdirectory_fails_before_upload(
    tmp_path: Path, upload_temporary_root: Path, asynchronous: bool, directory: str, ignore_patterns: str | None
) -> None:
    source = tmp_path / "source"
    blocked = source / directory
    blocked.mkdir(parents=True)
    (blocked / "evidence").write_bytes(b"must not silently disappear")
    (source / "summary.json").write_bytes(b"summary")
    blocked.chmod(0)
    fs = _RecordingFileSystem()
    try:
        with pytest.raises(PermissionError):
            await _upload(_manager(asynchronous, fs), source, ignore_patterns)
    finally:
        blocked.chmod(0o700)
    assert not fs.local_paths
    assert not list(upload_temporary_root.iterdir())


@pytest.mark.parametrize("asynchronous", [False, True], ids=["sync", "async"])
@pytest.mark.parametrize("error_number", [errno.ENOSPC, errno.EACCES], ids=["disk-full", "permission"])
async def test_copy_errors_fail_without_upload_or_reacquisition(
    tmp_path: Path,
    upload_temporary_root: Path,
    monkeypatch: pytest.MonkeyPatch,
    asynchronous: bool,
    error_number: int,
) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (source / "evidence").write_bytes(b"evidence")
    error = OSError(error_number, "copy failed")
    attempts = 0

    def fail_copy(local: Path, destination: Path) -> None:
        nonlocal attempts
        attempts += 1
        raise error

    monkeypatch.setattr(file_manager, "shutil", SimpleNamespace(copyfile=fail_copy, rmtree=shutil.rmtree))
    fs = _RecordingFileSystem()
    with pytest.raises(OSError) as raised:
        await _upload(_manager(asynchronous, fs), source)
    assert raised.value is error
    assert attempts == 1
    assert not fs.local_paths
    assert not list(upload_temporary_root.iterdir())


@pytest.mark.parametrize("asynchronous", [False, True], ids=["sync", "async"])
async def test_valid_empty_directory_and_file_symlink_selection(
    tmp_path: Path, upload_temporary_root: Path, asynchronous: bool
) -> None:
    source = tmp_path / "source"
    source.mkdir()
    fs = _RecordingFileSystem()
    manager = _manager(asynchronous, fs)
    await _upload(manager, source)
    assert not fs.uploads

    external = tmp_path / "external"
    external.mkdir()
    (external / "evidence").write_bytes(b"linked file")
    (source / "file-link").symlink_to(external / "evidence")
    (source / "directory-link").symlink_to(external, target_is_directory=True)
    await _upload(manager, source)
    assert fs.uploads == {"default/results#bundle/file-link": b"linked file"}
    assert not list(upload_temporary_root.iterdir())


@pytest.mark.parametrize("worker_fails", [False, True], ids=["worker-succeeds", "worker-fails"])
async def test_cancelled_acquisition_drains_worker_before_cleanup(
    tmp_path: Path, upload_temporary_root: Path, monkeypatch: pytest.MonkeyPatch, worker_fails: bool
) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (source / "evidence").write_bytes(b"evidence")
    started, release, finished = Event(), Event(), Event()
    acquire = file_manager._stage_local_files

    def blocked_acquisition(
        local_path: Path, staging_root: Path, ignore_patterns: list[str] | str | None
    ) -> tuple[Path, list[str]]:
        started.set()
        try:
            if not release.wait(timeout=5):
                raise TimeoutError("Test did not release acquisition")
            if worker_fails:
                raise OSError("worker failed after cancellation")
            return acquire(local_path, staging_root, ignore_patterns)
        finally:
            finished.set()

    monkeypatch.setattr(file_manager, "_stage_local_files", blocked_acquisition)
    fs = _RecordingFileSystem()
    loop = asyncio.get_running_loop()
    previous_handler = loop.get_exception_handler()
    unhandled: list[dict[str, object]] = []
    loop.set_exception_handler(lambda _loop, context: unhandled.append(context))
    task = asyncio.create_task(_upload(_manager(True, fs), source))
    try:
        assert await asyncio.to_thread(started.wait, 5)
        task.cancel("first cancellation")
        await asyncio.sleep(0)
        task.cancel("second cancellation")
        await asyncio.sleep(0)
        assert not task.done(), "The acquisition worker still owns the temporary directory"
        assert list(upload_temporary_root.iterdir())
        release.set()
        with pytest.raises(asyncio.CancelledError, match="first cancellation"):
            await asyncio.wait_for(task, timeout=5)
        assert finished.is_set()
        assert not fs.local_paths
        assert not list(upload_temporary_root.iterdir())
        await asyncio.sleep(0)
        assert not unhandled
    finally:
        release.set()
        assert await asyncio.to_thread(finished.wait, 5)
        await asyncio.gather(task, return_exceptions=True)
        loop.set_exception_handler(previous_handler)


@pytest.mark.parametrize("asynchronous", [False, True], ids=["sync", "async"])
async def test_upload_error_preserves_origin_without_reacquiring(
    tmp_path: Path, upload_temporary_root: Path, monkeypatch: pytest.MonkeyPatch, asynchronous: bool
) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (source / "evidence").write_bytes(b"evidence")
    error = OSError("remote upload failed")
    fs = _RecordingFileSystem()
    list_files = file_manager._list_selected_files
    listings = 0

    def count_listing(path: Path, ignore_patterns: list[str] | str | None = None) -> list[str]:
        nonlocal listings
        listings += 1
        return list_files(path, ignore_patterns=ignore_patterns)

    def fail_upload(local: str, remote: str) -> None:
        fs.local_paths.append(Path(local))
        raise error

    monkeypatch.setattr(file_manager, "_list_selected_files", count_listing)
    monkeypatch.setattr(fs, "put_file", fail_upload)
    with pytest.raises(OSError) as raised:
        await _upload(_manager(asynchronous, fs), source)
    assert raised.value is error
    assert listings == 1
    assert len(fs.local_paths) == 1
    assert any(
        str(source / "evidence") in note and "default/results#bundle/evidence" in note
        for note in getattr(error, "__notes__", [])
    )
    assert not list(upload_temporary_root.iterdir())


@pytest.mark.parametrize("asynchronous", [False, True], ids=["sync", "async"])
async def test_upload_cancellation_stops_and_cleans_staging(
    tmp_path: Path, upload_temporary_root: Path, monkeypatch: pytest.MonkeyPatch, asynchronous: bool
) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (source / "first").write_bytes(b"first")
    (source / "second").write_bytes(b"second")
    fs = _RecordingFileSystem()
    cancellation = asyncio.CancelledError("stop uploads")

    def cancel_upload(local: str, remote: str) -> None:
        fs.local_paths.append(Path(local))
        raise cancellation

    monkeypatch.setattr(fs, "put_file", cancel_upload)
    with pytest.raises(asyncio.CancelledError) as raised:
        await _upload(_manager(asynchronous, fs), source)
    assert raised.value is cancellation
    assert len(fs.local_paths) == 1
    assert not list(upload_temporary_root.iterdir())


@pytest.mark.parametrize("asynchronous", [False, True], ids=["sync", "async"])
async def test_failed_upload_is_not_registered_as_complete(
    tmp_path: Path, upload_temporary_root: Path, monkeypatch: pytest.MonkeyPatch, asynchronous: bool
) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (source / "evidence").write_bytes(b"evidence")
    fs = _RecordingFileSystem()
    error = OSError("remote upload failed")

    def fail_upload(local: str, remote: str) -> None:
        raise error

    monkeypatch.setattr(fs, "put_file", fail_upload)
    file_sink = _manager(asynchronous, fs)
    response = Mock()
    response.data.return_value = SimpleNamespace(fileset="results", output_location=None)
    if asynchronous:
        jobs_client = Mock(spec=AsyncJobsClient)
        jobs_client.get_job = AsyncMock(return_value=response)
        manager = AsyncResultManager(
            job_name="job",
            workspace="default",
            attempt_id="attempt",
            files_client=Mock(spec=AsyncFilesClient),
            jobs_client=jobs_client,
        )
    else:
        jobs_client = Mock(spec=JobsClient)
        jobs_client.get_job.return_value = response
        manager = ResultManager(
            job_name="job",
            workspace="default",
            attempt_id="attempt",
            files_client=Mock(spec=FilesClient),
            jobs_client=jobs_client,
        )
    monkeypatch.setattr(manager, "_create_file_manager", lambda _name: file_sink)
    with pytest.raises(OSError) as raised:
        if isinstance(manager, AsyncResultManager):
            await manager.create_result("bundle", source, stage=True)
        else:
            manager.create_result("bundle", source, stage=True)
    assert raised.value is error
    jobs_client.create_job_result.assert_not_called()
    assert not list(upload_temporary_root.iterdir())


@pytest.fixture
def visited_directories(monkeypatch: pytest.MonkeyPatch) -> set[Path]:
    visited: set[Path] = set()
    walk = file_manager.walk

    def record_walk(
        path: Path, *, onerror: Callable[[OSError], object], followlinks: bool
    ) -> Iterator[tuple[str, list[str], list[str]]]:
        for entry in walk(path, onerror=onerror, followlinks=followlinks):
            visited.add(Path(entry[0]))
            # Stream the original entry, retaining the directory list the real walk consults.
            yield entry

    monkeypatch.setattr(file_manager, "walk", record_walk)
    return visited


@pytest.mark.parametrize("asynchronous", [False, True], ids=["sync", "async"])
@pytest.mark.parametrize("directory", ["cache", "nested/cache"], ids=["root", "nested"])
@pytest.mark.parametrize("ignore_patterns", ["cache/", ["cache/"]], ids=["string", "list"])
@pytest.mark.skipif(os.geteuid() == 0, reason="Root bypasses ordinary directory permissions")
async def test_ignored_unreadable_directory_does_not_block_upload(
    tmp_path: Path,
    upload_temporary_root: Path,
    asynchronous: bool,
    directory: str,
    ignore_patterns: list[str] | str,
) -> None:
    source = tmp_path / "source"
    blocked = source / directory
    blocked.mkdir(parents=True)
    (blocked / "scratch").write_bytes(b"excluded")
    (source / "summary.json").write_bytes(b"summary")
    fs = _RecordingFileSystem()
    blocked.chmod(0)
    try:
        await _upload(_manager(asynchronous, fs), source, ignore_patterns)
    finally:
        blocked.chmod(0o700)

    assert fs.uploads == {"default/results#bundle/summary.json": b"summary"}
    assert not list(upload_temporary_root.iterdir())


@pytest.mark.parametrize("asynchronous", [False, True], ids=["sync", "async"])
@pytest.mark.parametrize(("pattern", "directory"), [("cache", "cache"), ("*.tmp", "cache.tmp")])
async def test_file_patterns_do_not_prune_directories(
    tmp_path: Path, upload_temporary_root: Path, asynchronous: bool, pattern: str, directory: str
) -> None:
    source = tmp_path / "source"
    (source / directory).mkdir(parents=True)
    (source / directory / "keep.txt").write_bytes(b"selected")
    (source / "summary.json").write_bytes(b"summary")
    (source / "excluded").mkdir()
    (source / "excluded" / directory).write_bytes(b"file-pattern match")
    fs = _RecordingFileSystem()

    await _upload(_manager(asynchronous, fs), source, pattern)

    assert fs.uploads == {
        "default/results#bundle/summary.json": b"summary",
        f"default/results#bundle/{directory}/keep.txt": b"selected",
    }
    assert not list(upload_temporary_root.iterdir())


@pytest.mark.parametrize("asynchronous", [False, True], ids=["sync", "async"])
async def test_directory_pattern_does_not_exclude_upload_root(
    tmp_path: Path, upload_temporary_root: Path, asynchronous: bool
) -> None:
    source = tmp_path / "cache"
    (source / "cache").mkdir(parents=True)
    (source / "cache/scratch").write_bytes(b"excluded child")
    (source / "summary.json").write_bytes(b"selected root")
    fs = _RecordingFileSystem()

    await _upload(_manager(asynchronous, fs), source, "cache/")

    assert fs.uploads == {"default/results#bundle/summary.json": b"selected root"}
    assert not list(upload_temporary_root.iterdir())


@pytest.mark.parametrize("asynchronous", [False, True], ids=["sync", "async"])
async def test_directory_upload_does_not_visit_excluded_subtrees(
    tmp_path: Path, upload_temporary_root: Path, visited_directories: set[Path], asynchronous: bool
) -> None:
    source = tmp_path / "source"
    for directory in ("cache", "nested/cache", "selected"):
        (source / directory).mkdir(parents=True)
        (source / directory / "keep.txt").write_bytes(b"payload")
    fs = _RecordingFileSystem()

    await _upload(_manager(asynchronous, fs), source, ["cache/"])

    assert fs.uploads == {"default/results#bundle/selected/keep.txt": b"payload"}
    assert source / "selected" in visited_directories
    assert visited_directories.isdisjoint({source / "cache", source / "nested/cache"})
    assert not list(upload_temporary_root.iterdir())


@pytest.mark.parametrize("asynchronous", [False, True], ids=["sync", "async"])
async def test_ignored_directory_removed_during_walk_does_not_retry(
    tmp_path: Path,
    upload_temporary_root: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    asynchronous: bool,
) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (source / "summary.json").write_bytes(b"summary")
    walk = file_manager.walk
    acquisitions = 0

    def remove_after_parent_listing(
        path: Path, *, onerror: Callable[[OSError], object], followlinks: bool
    ) -> Iterator[tuple[str, list[str], list[str]]]:
        nonlocal acquisitions
        acquisitions += 1
        cache = path / "cache"
        cache.mkdir()
        for entry in walk(path, onerror=onerror, followlinks=followlinks):
            yield entry
            if Path(entry[0]) == path:
                # External cleanup still happens if the caller pruned cache from entry[1].
                cache.rmdir()

    monkeypatch.setattr(file_manager, "walk", remove_after_parent_listing)
    caplog.set_level(logging.WARNING, logger=file_manager.__name__)
    fs = _RecordingFileSystem()

    await _upload(_manager(asynchronous, fs), source, ["cache/"])

    assert fs.uploads == {"default/results#bundle/summary.json": b"summary"}
    assert acquisitions == 1
    assert not [record for record in caplog.records if record.name == file_manager.__name__]
    assert not list(upload_temporary_root.iterdir())


@pytest.mark.parametrize("asynchronous", [False, True], ids=["sync", "async"])
@pytest.mark.parametrize(
    ("pattern", "excluded", "kept"),
    [
        (
            "plugins*/",
            {"plugins", "nested/plugins-clone", "home/.tmp/plugins-clone", "deep/home/.tmp/plugins-clone"},
            {"summary.json", "plugins.sha"},
        ),
        (
            "home/.tmp/plugins*/",
            {"home/.tmp/plugins-clone"},
            {
                "summary.json",
                "plugins.sha",
                "plugins/root",
                "nested/plugins-clone/nested",
                "deep/home/.tmp/plugins-clone/deep",
            },
        ),
        (
            "/home/.tmp/plugins*/",
            {"home/.tmp/plugins-clone"},
            {
                "summary.json",
                "plugins.sha",
                "plugins/root",
                "nested/plugins-clone/nested",
                "deep/home/.tmp/plugins-clone/deep",
            },
        ),
    ],
)
async def test_directory_globs_prune_only_matching_relative_directories(
    tmp_path: Path,
    upload_temporary_root: Path,
    visited_directories: set[Path],
    asynchronous: bool,
    pattern: str,
    excluded: set[str],
    kept: set[str],
) -> None:
    source = tmp_path / "source"
    files = {
        "summary.json",
        "plugins.sha",
        "plugins/root",
        "nested/plugins-clone/nested",
        "home/.tmp/plugins-clone/rooted",
        "deep/home/.tmp/plugins-clone/deep",
    }
    for relative in files:
        path = source / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(relative.encode())
    fs = _RecordingFileSystem()

    await _upload(_manager(asynchronous, fs), source, [pattern])

    assert fs.uploads == {f"default/results#bundle/{relative}": relative.encode() for relative in kept}
    assert visited_directories.isdisjoint({source / relative for relative in excluded})
    assert {source / Path(relative).parent for relative in kept}.issubset(visited_directories)
    assert not list(upload_temporary_root.iterdir())


@pytest.mark.parametrize("asynchronous", [False, True], ids=["sync", "async"])
@pytest.mark.parametrize("pattern", ["home/.tmp/plugins*/", "/home/.tmp/plugins*/"])
@pytest.mark.skipif(os.geteuid() == 0, reason="Root bypasses ordinary directory permissions")
async def test_anchored_directory_patterns_preserve_deeper_traversal_errors(
    tmp_path: Path, upload_temporary_root: Path, asynchronous: bool, pattern: str
) -> None:
    source = tmp_path / "source"
    blocked = source / "deep/home/.tmp/plugins-clone"
    blocked.mkdir(parents=True)
    (blocked / "evidence").write_bytes(b"selected")
    (source / "summary.json").write_bytes(b"summary")
    fs = _RecordingFileSystem()
    blocked.chmod(0)
    try:
        with pytest.raises(PermissionError):
            await _upload(_manager(asynchronous, fs), source, [pattern])
    finally:
        blocked.chmod(0o700)

    assert not fs.local_paths
    assert not list(upload_temporary_root.iterdir())


@pytest.mark.parametrize("asynchronous", [False, True], ids=["sync", "async"])
async def test_unstaged_directory_upload_reads_the_source_directly(
    tmp_path: Path, upload_temporary_root: Path, monkeypatch: pytest.MonkeyPatch, asynchronous: bool
) -> None:
    source = tmp_path / "source"
    (source / "nested").mkdir(parents=True)
    (source / "summary.json").write_bytes(b"summary")
    (source / "nested/evidence").write_bytes(b"evidence")
    (source / "debug.log").write_bytes(b"ignored")

    def unexpected_staging(*_args: object, **_kwargs: object) -> NoReturn:
        raise AssertionError("an unstaged upload must not stage")

    monkeypatch.setattr(file_manager, "_stage_local_files", unexpected_staging)
    fs = _RecordingFileSystem()

    await _upload(_manager(asynchronous, fs), source, "*.log", stage=False)

    assert fs.uploads == {
        "default/results#bundle/summary.json": b"summary",
        "default/results#bundle/nested/evidence": b"evidence",
    }
    assert sorted(fs.local_paths) == sorted([source / "summary.json", source / "nested/evidence"])
    assert not list(upload_temporary_root.iterdir())
