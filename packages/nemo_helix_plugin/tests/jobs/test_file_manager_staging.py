# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Directory upload acquisition uses real files, including a disappearing clone lock."""

import os
from pathlib import Path

import pytest
from nemo_helix_plugin.jobs import file_manager


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
    list_files = file_manager._list_local_files_sync
    listings = 0

    def list_then_finish_clone(path: Path) -> list[str]:
        nonlocal listings
        files = list_files(path)
        listings += 1
        if listings == 1:
            lock.unlink()
        return files

    # Schedule a real unlink at the traversal/open boundary, without replacing
    # pathlib, shutil, open, or any product service/client.
    monkeypatch.setattr(file_manager, "_list_local_files_sync", list_then_finish_clone)
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
    list_files = file_manager._list_local_files_sync
    listings = 0

    def list_then_remove_required(path: Path) -> list[str]:
        nonlocal listings
        required.write_text("required trial")
        files = list_files(path)
        listings += 1
        required.unlink()
        return files

    monkeypatch.setattr(file_manager, "_list_local_files_sync", list_then_remove_required)
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
