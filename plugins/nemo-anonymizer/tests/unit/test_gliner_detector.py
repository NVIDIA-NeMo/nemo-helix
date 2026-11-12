# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os
from collections.abc import Iterator
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
    monkeypatch.setattr(gliner_detector, "_hf_hub_cache_dir", lambda: tmp_path / "hub")
    assert gliner_detector.is_gliner_cached() is False

    snapshot = (
        tmp_path
        / "hub"
        / repo_folder_name(repo_id=gliner_detector.GLINER_MODEL_ID, repo_type="model")
        / "snapshots"
        / gliner_detector.GLINER_MODEL_REVISION
    )
    snapshot.mkdir(parents=True)
    # A config-only snapshot is a partial/interrupted download, not a usable model.
    (snapshot / "config.json").write_text("{}")
    assert gliner_detector.is_gliner_cached() is False
    # Weights alongside the config make it complete.
    (snapshot / "model.safetensors").write_bytes(b"\x00")
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
    # Isolate the HuggingFace cache to tmp_path so the test never touches the real one.
    hub = tmp_path / "hub"
    monkeypatch.setattr(gliner_detector, "_hf_hub_cache_dir", lambda: hub)
    monkeypatch.setattr(gliner_detector, "is_gliner_cached", lambda: False)

    fileset_folder = hub / repo_folder_name(
        repo_id=f"{gliner_detector.GLINER_FILESET_WORKSPACE}/{gliner_detector.GLINER_FILESET_NAME}",
        repo_type="model",
    )

    captured: dict[str, Any] = {}

    def _fake_download(**kwargs: Any) -> str:
        # Mimic a Files pull-through: Hub writes a full repo-cache folder
        # (snapshots/<files-hash> + refs/<revision> + blobs).
        captured.update(kwargs)
        files_hash = "f" * 40
        snapshot = fileset_folder / "snapshots" / files_hash
        snapshot.mkdir(parents=True)
        (snapshot / "config.json").write_text("{}")
        (snapshot / "model.safetensors").write_bytes(b"\x00")
        (fileset_folder / "refs").mkdir(parents=True, exist_ok=True)
        (fileset_folder / "refs" / kwargs["revision"]).write_text(files_hash)
        (fileset_folder / "blobs").mkdir(parents=True, exist_ok=True)
        return str(snapshot)

    monkeypatch.setattr(gliner_detector, "snapshot_download", _fake_download)
    started = Mock()

    gliner_detector.prewarm_gliner_cache("http://localhost:8080/", on_download_start=started)

    started.assert_called_once()
    # Pulled through the Files HF resolve API as the fileset ref, with the service token.
    assert captured["repo_id"] == f"{gliner_detector.GLINER_FILESET_WORKSPACE}/{gliner_detector.GLINER_FILESET_NAME}"
    assert captured["endpoint"] == "http://localhost:8080/apis/files/v2/hf"
    assert captured["revision"] == gliner_detector.GLINER_MODEL_REVISION
    assert captured["token"] == "service:anonymizer"

    # The whole repo-cache folder is COPIED (not symlinked) to the upstream repo name,
    # with the Hub metadata intact so upstream's offline snapshot_download resolves it.
    upstream_folder = hub / repo_folder_name(repo_id=gliner_detector.GLINER_MODEL_ID, repo_type="model")
    assert upstream_folder.is_dir() and not upstream_folder.is_symlink()
    assert (upstream_folder / "refs" / gliner_detector.GLINER_MODEL_REVISION).read_text() == "f" * 40
    copied_snapshot = upstream_folder / "snapshots" / ("f" * 40)
    assert (copied_snapshot / "config.json").read_text() == "{}"
    assert (copied_snapshot / "model.safetensors").exists()

    # Files named the snapshot after its OWN commit hash ("f"*40), but upstream
    # resolves revision=<pinned SHA> by looking for snapshots/<SHA> directly, so
    # the align must also expose a snapshots/<SHA> entry pointing at it. Without
    # this the offline loader fails with LocalEntryNotFoundError (regression:
    # the real Files pull-through uses a hash != the pinned SHA).
    sha_snapshot = upstream_folder / "snapshots" / gliner_detector.GLINER_MODEL_REVISION
    assert sha_snapshot.is_symlink()
    assert (sha_snapshot / "config.json").read_text() == "{}"
    assert (sha_snapshot / "model.safetensors").exists()


class _Overrides:
    def __init__(self, detection: dict[str, object] | None) -> None:
        self.detection = detection


@pytest.mark.parametrize(
    ("selected_models", "expected"),
    [
        (None, False),
        (_Overrides(None), False),
        (_Overrides({}), False),
        (_Overrides({"entity_validator": "gpt"}), False),
        (_Overrides({"entity_detector": "gliner-pii-detector"}), True),
    ],
)
def test_caller_supplied_entity_detector(selected_models: object | None, expected: bool) -> None:
    assert gliner_detector.caller_supplied_entity_detector(selected_models) is expected


@pytest.fixture
def read_only_dir(tmp_path: Path) -> Iterator[Path]:
    """A directory the test process cannot write into, like a hardened pod's ``$HOME``."""
    if os.geteuid() == 0:
        pytest.skip("root ignores directory permissions")
    path = tmp_path / "read-only-home"
    path.mkdir()
    path.chmod(0o555)
    yield path
    path.chmod(0o755)


_CACHE_ENV_VARS = ("HF_HOME", "HF_HUB_CACHE", "XDG_CACHE_HOME")


def test_gliner_caches_under_redirects_unwritable_caches_to_scratch(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, read_only_dir: Path
) -> None:
    monkeypatch.delenv("HF_HOME", raising=False)
    monkeypatch.setenv("HF_HUB_CACHE", str(read_only_dir / ".cache" / "huggingface" / "hub"))
    monkeypatch.setenv("XDG_CACHE_HOME", str(read_only_dir / ".cache"))
    before = {name: os.environ.get(name) for name in _CACHE_ENV_VARS}
    scratch = tmp_path / "scratch"

    with gliner_detector.gliner_caches_under(scratch):
        assert os.environ["HF_HOME"] == str(scratch / "huggingface")
        assert os.environ["HF_HUB_CACHE"] == str(scratch / "huggingface" / "hub")
        assert os.environ["XDG_CACHE_HOME"] == str(scratch / "cache")
        # The prewarm and the GLiNER child must agree on the cache location.
        assert gliner_detector._hf_hub_cache_dir() == scratch / "huggingface" / "hub"
        assert gliner_detector._upstream_snapshot_dir().is_relative_to(scratch)

    assert {name: os.environ.get(name) for name in _CACHE_ENV_VARS} == before


def test_gliner_caches_under_keeps_writable_caches(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    hub = tmp_path / "home" / ".cache" / "huggingface" / "hub"  # not created yet; parent is writable
    monkeypatch.setenv("HF_HUB_CACHE", str(hub))
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "home" / ".cache"))
    monkeypatch.delenv("HF_HOME", raising=False)

    with gliner_detector.gliner_caches_under(tmp_path / "scratch"):
        assert os.environ["HF_HUB_CACHE"] == str(hub)
        assert os.environ["XDG_CACHE_HOME"] == str(tmp_path / "home" / ".cache")
        assert "HF_HOME" not in os.environ


def test_gliner_caches_under_restores_environment_on_error(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, read_only_dir: Path
) -> None:
    monkeypatch.setenv("HF_HUB_CACHE", str(read_only_dir / "hub"))
    monkeypatch.setenv("XDG_CACHE_HOME", str(read_only_dir))
    monkeypatch.delenv("HF_HOME", raising=False)

    with pytest.raises(RuntimeError, match="boom"):
        with gliner_detector.gliner_caches_under(tmp_path / "scratch"):
            raise RuntimeError("boom")

    assert os.environ["HF_HUB_CACHE"] == str(read_only_dir / "hub")
    assert os.environ["XDG_CACHE_HOME"] == str(read_only_dir)
    assert "HF_HOME" not in os.environ
