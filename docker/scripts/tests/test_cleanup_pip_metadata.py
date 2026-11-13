# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import importlib.util
import os
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).parents[1] / "cleanup-pip-metadata.py"
SPEC = importlib.util.spec_from_file_location("cleanup_pip_metadata", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
cleanup = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(cleanup)


def metadata(root: Path, version: str) -> Path:
    directory = root / f"pip-{version}.dist-info"
    directory.mkdir(parents=True)
    (directory / "METADATA").write_text(f"Name: pip\nVersion: {version}\n")
    return directory


@pytest.mark.parametrize("code_version", ["26.2.2", "26.3"])
def test_final_cleanup_matches_imported_code_instead_of_choosing_newest(tmp_path: Path, code_version: str) -> None:
    package = tmp_path / "pip"
    package.mkdir()
    source = package / "__init__.py"
    source.write_text(f"__version__ = {code_version!r}\n")
    for version in ("26.2.2", "26.3"):
        metadata(tmp_path, version)

    result = subprocess.run(
        [sys.executable, "-S", str(SCRIPT)],
        env={**os.environ, "PYTHONPATH": str(tmp_path)},
        capture_output=True,
        text=True,
        check=True,
    )

    assert list(tmp_path.glob("pip-*.dist-info")) == [tmp_path / f"pip-{code_version}.dist-info"]
    assert source.read_text() == f"__version__ = {code_version!r}\n"
    assert f"Validated exactly one pip distribution: {code_version}" in result.stdout


def test_cleanup_is_idempotent_for_a_consistent_distribution(tmp_path: Path) -> None:
    keep = metadata(tmp_path, "26.2.2")

    cleanup.cleanup_pip_metadata(tmp_path, "26.2.2")
    cleanup.cleanup_pip_metadata(tmp_path, "26.2.2")

    assert list(tmp_path.glob("pip-*.dist-info")) == [keep]


@pytest.mark.parametrize("matching_count", [0, 2])
def test_cleanup_fails_before_deleting_when_matching_metadata_is_ambiguous(tmp_path: Path, matching_count: int) -> None:
    stale = metadata(tmp_path, "26.3")
    if matching_count:
        metadata(tmp_path, "26.2.2")
        duplicate = tmp_path / "pip-26.2.2-duplicate.dist-info"
        duplicate.mkdir()
        (duplicate / "METADATA").write_text("Name: pip\nVersion: 26.2.2\n")
    before = sorted(tmp_path.glob("pip-*.dist-info"))

    with pytest.raises(RuntimeError, match="Expected one pip metadata directory matching code"):
        cleanup.cleanup_pip_metadata(tmp_path, "26.2.2")

    assert sorted(tmp_path.glob("pip-*.dist-info")) == before
    assert stale.exists()


def test_cleanup_unlinks_stale_metadata_without_deleting_shared_cache_target(tmp_path: Path) -> None:
    environment = tmp_path / "site-packages"
    keep = metadata(environment, "26.2.2")
    cached = metadata(tmp_path / "cache", "26.3")
    stale = environment / cached.name
    stale.symlink_to(cached, target_is_directory=True)

    cleanup.cleanup_pip_metadata(environment, "26.2.2")

    assert list(environment.glob("pip-*.dist-info")) == [keep]
    assert cached.exists()
    assert not stale.is_symlink()
