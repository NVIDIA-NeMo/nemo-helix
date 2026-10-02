# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""How a local storage path is resolved, and which state it keeps together.

Regression: ``NHX_DATA_DIR`` relocates the entity-store database, but the bundled local
configurations pinned the literal blob location, so a run with it set produced a *half*
isolated instance — database in the chosen directory, blobs in the default one. That is worse
than an un-isolated instance: it looks isolated, and wiping the chosen directory silently
leaves the blobs behind.
"""

import pytest
from nemo_helix_plugin.files.storage_config import LocalStorageConfig


def test_an_empty_path_follows_the_platform_data_dir(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    """The fix: blobs land under the chosen data directory, beside the entity-store database."""
    monkeypatch.setenv("NHX_DATA_DIR", str(tmp_path))
    assert LocalStorageConfig(path="").path == str(tmp_path / "files")


def test_an_empty_path_without_a_data_dir_keeps_the_previous_location(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The value the bundled configs used to hard-code. Anyone who has not opted into a data
    directory must see no change at all."""
    monkeypatch.delenv("NHX_DATA_DIR", raising=False)
    monkeypatch.setenv("HOME", "/home/someone")
    monkeypatch.delenv("XDG_DATA_HOME", raising=False)
    assert LocalStorageConfig(path="").path == "/home/someone/.local/share/nemo/files"


def test_an_explicit_path_is_never_overridden(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    """The container image, the Helm chart and the agentic runners all set this explicitly."""
    monkeypatch.setenv("NHX_DATA_DIR", str(tmp_path))
    assert LocalStorageConfig(path="/data/files_storage").path == "/data/files_storage"


def test_home_relative_paths_still_expand(monkeypatch: pytest.MonkeyPatch) -> None:
    """Only the empty path gains meaning; every other form resolves exactly as before."""
    monkeypatch.setenv("HOME", "/home/someone")
    assert LocalStorageConfig(path="~/blobs").path == "/home/someone/blobs"


def test_the_data_dir_is_read_per_construction(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    """Resolution happens in the validator, not as a field default, so it tracks the
    environment at construction — and, just as importantly, keeps a machine-dependent path out
    of the field default that generates the committed config reference."""
    monkeypatch.setenv("NHX_DATA_DIR", str(tmp_path / "one"))
    assert LocalStorageConfig(path="").path == str(tmp_path / "one" / "files")
    monkeypatch.setenv("NHX_DATA_DIR", str(tmp_path / "two"))
    assert LocalStorageConfig(path="").path == str(tmp_path / "two" / "files")
