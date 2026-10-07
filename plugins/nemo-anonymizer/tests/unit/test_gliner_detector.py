# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import Mock

import httpx
import pytest
from huggingface_hub.file_download import repo_folder_name
from nemo_anonymizer_plugin.app import gliner_detector
from nemo_anonymizer_plugin.app.errors import AnonymizerInternalError
from nemo_helix_plugin.client.errors import NotFoundError


class _Storage:
    def __init__(self, type_: str, repo_id: str | None) -> None:
        self.type = type_
        self.repo_id = repo_id


class _Fileset:
    def __init__(self, storage: _Storage) -> None:
        self.storage = storage


class _FakeFilesClient:
    """Minimal stand-in for the FilesClient get/create_fileset surface."""

    def __init__(self, *, existing: _Fileset | None = None) -> None:
        self._existing = existing
        self.created: list[dict[str, Any]] = []

    def get_fileset(self, *, name: str, workspace: str) -> _Fileset:
        if self._existing is None:
            raise NotFoundError(httpx.Response(status_code=404, text="not found"))
        return self._existing

    def create_fileset(self, *, workspace: str, body: Any) -> _Fileset:
        self.created.append({"workspace": workspace, "body": body})
        self._existing = _Fileset(_Storage("huggingface", gliner_detector.GLINER_MODEL_ID))
        return self._existing


def _patch_files_client(monkeypatch: pytest.MonkeyPatch, fake: _FakeFilesClient) -> None:
    monkeypatch.setattr(gliner_detector, "client_from_platform", lambda sdk, client_cls: fake)


def test_ensure_fileset_creates_when_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _FakeFilesClient(existing=None)
    _patch_files_client(monkeypatch, fake)

    gliner_detector.ensure_gliner_fileset(Mock())

    assert len(fake.created) == 1
    body = fake.created[0]["body"]
    assert body.name == gliner_detector.GLINER_FILESET_NAME
    assert fake.created[0]["workspace"] == gliner_detector.GLINER_FILESET_WORKSPACE
    assert body.storage.repo_id == gliner_detector.GLINER_MODEL_ID
    assert body.storage.revision == gliner_detector.GLINER_MODEL_REVISION


def test_ensure_fileset_is_idempotent_when_present_and_matching(monkeypatch: pytest.MonkeyPatch) -> None:
    existing = _Fileset(_Storage("huggingface", gliner_detector.GLINER_MODEL_ID))
    fake = _FakeFilesClient(existing=existing)
    _patch_files_client(monkeypatch, fake)

    gliner_detector.ensure_gliner_fileset(Mock())

    assert fake.created == []


def test_ensure_fileset_rejects_mismatched_existing_fileset(monkeypatch: pytest.MonkeyPatch) -> None:
    # A same-named fileset that points at a different repo must fail loudly.
    existing = _Fileset(_Storage("huggingface", "someone-else/not-gliner"))
    fake = _FakeFilesClient(existing=existing)
    _patch_files_client(monkeypatch, fake)

    with pytest.raises(AnonymizerInternalError, match="not the expected"):
        gliner_detector.ensure_gliner_fileset(Mock())


def test_ensure_fileset_rejects_wrong_storage_type(monkeypatch: pytest.MonkeyPatch) -> None:
    existing = _Fileset(_Storage("local", None))
    fake = _FakeFilesClient(existing=existing)
    _patch_files_client(monkeypatch, fake)

    with pytest.raises(AnonymizerInternalError, match="not the expected"):
        gliner_detector.ensure_gliner_fileset(Mock())


def test_is_gliner_cached_reflects_snapshot_presence(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("HF_HOME", str(tmp_path))
    assert gliner_detector.is_gliner_cached() is False

    snapshot = (
        tmp_path
        / "hub"
        / repo_folder_name(repo_id=gliner_detector.GLINER_MODEL_ID, repo_type="model")
        / "snapshots"
        / gliner_detector.GLINER_MODEL_REVISION
    )
    snapshot.mkdir(parents=True)
    (snapshot / "config.json").write_text("{}")
    assert gliner_detector.is_gliner_cached() is True


def test_prewarm_is_noop_when_cached(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(gliner_detector, "is_gliner_cached", lambda: True)
    started = Mock()

    def _should_not_download(**kwargs: object) -> str:
        raise AssertionError("snapshot_download must not run when weights are cached")

    monkeypatch.setattr(gliner_detector, "snapshot_download", _should_not_download)

    gliner_detector.prewarm_gliner_cache("http://localhost:8080", on_download_start=started)

    started.assert_not_called()


def test_prewarm_downloads_through_files_and_aligns_cache(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("HF_HOME", str(tmp_path))
    monkeypatch.setattr(gliner_detector, "is_gliner_cached", lambda: False)

    # Simulate the Files pull-through landing a snapshot under the fileset's cache folder.
    fileset_snapshot = (
        tmp_path
        / "hub"
        / repo_folder_name(
            repo_id=f"{gliner_detector.GLINER_FILESET_WORKSPACE}/{gliner_detector.GLINER_FILESET_NAME}",
            repo_type="model",
        )
        / "snapshots"
        / "files-generated-commit-hash"
    )
    fileset_snapshot.mkdir(parents=True)
    (fileset_snapshot / "config.json").write_text("{}")

    captured: dict[str, Any] = {}

    def _fake_download(**kwargs: Any) -> str:
        captured.update(kwargs)
        return str(fileset_snapshot)

    monkeypatch.setattr(gliner_detector, "snapshot_download", _fake_download)
    started = Mock()

    gliner_detector.prewarm_gliner_cache("http://localhost:8080/", on_download_start=started)

    started.assert_called_once()
    # Pulled through the Files HF resolve API as the fileset ref, with the service token.
    assert captured["repo_id"] == f"{gliner_detector.GLINER_FILESET_WORKSPACE}/{gliner_detector.GLINER_FILESET_NAME}"
    assert captured["endpoint"] == "http://localhost:8080/apis/files/v2/hf"
    assert captured["revision"] == gliner_detector.GLINER_MODEL_REVISION
    # The result is aligned to the repo id/revision upstream reads.
    upstream_snapshot = (
        tmp_path
        / "hub"
        / repo_folder_name(repo_id=gliner_detector.GLINER_MODEL_ID, repo_type="model")
        / "snapshots"
        / gliner_detector.GLINER_MODEL_REVISION
    )
    assert upstream_snapshot.is_symlink()
    assert (upstream_snapshot / "config.json").read_text() == "{}"
