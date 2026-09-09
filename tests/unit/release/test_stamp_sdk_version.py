# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import importlib.util
import os
import subprocess
from pathlib import Path
from types import ModuleType

import pytest


def load_stamp_module() -> ModuleType:
    script_path = Path(__file__).parents[3] / ".github/scripts/stamp_sdk_version.py"
    spec = importlib.util.spec_from_file_location("stamp_sdk_version", script_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


stamp_sdk_version = load_stamp_module()
StampError = stamp_sdk_version.StampError


def git(source_root: Path, *args: str) -> None:
    subprocess.run(
        ["git", "-C", str(source_root), *args],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env={**os.environ, "GIT_CONFIG_GLOBAL": os.devnull, "GIT_CONFIG_NOSYSTEM": "1"},
    )


def commit(source_root: Path, message: str) -> None:
    marker = source_root / "marker.txt"
    marker.write_text(marker.read_text(encoding="utf-8") + f"{message}\n", encoding="utf-8")
    git(source_root, "add", "marker.txt")
    git(
        source_root,
        "-c",
        "user.name=Test User",
        "-c",
        "user.email=test@example.com",
        "-c",
        "commit.gpgsign=false",
        "commit",
        "-m",
        message,
    )


def init_git_source(source_root: Path) -> None:
    source_root.mkdir(parents=True)
    subprocess.run(
        ["git", "init", str(source_root)],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env={**os.environ, "GIT_CONFIG_GLOBAL": os.devnull, "GIT_CONFIG_NOSYSTEM": "1"},
    )
    (source_root / "marker.txt").write_text("", encoding="utf-8")
    commit(source_root, "initial")


def stamp(
    source_root: Path,
    *,
    sdk_id: str = "nemo-platform",
    cadence: str = "release",
    release_label: str = "1.0.0",
    nightly_timestamp: str = "",
) -> str:
    return stamp_sdk_version.stamp_sdk_version(
        source_root=source_root,
        sdk_id=sdk_id,
        cadence=cadence,
        release_label=release_label,
        nightly_timestamp=nightly_timestamp,
    )


def test_nightly_uses_latest_reachable_release_core_tag(tmp_path: Path):
    source_root = tmp_path / "source"
    init_git_source(source_root)
    git(source_root, "tag", "1.2.0")
    commit(source_root, "feature")
    git(source_root, "tag", "2.1.0-rc0")
    commit(source_root, "after-release")

    version = stamp(
        source_root,
        cadence="nightly",
        release_label="nightly-20260512010101",
        nightly_timestamp="20260512010101",
    )

    assert version == "2.1.0.dev20260512010101"


def test_nightly_falls_back_when_no_release_tag_is_available(tmp_path: Path):
    source_root = tmp_path / "source"
    init_git_source(source_root)
    git(source_root, "tag", "v9.9.9")
    git(source_root, "tag", "1.2.3rc0")

    version = stamp(
        source_root,
        cadence="nightly",
        release_label="nightly-20260512010101",
        nightly_timestamp="20260512010101",
    )

    assert version == "0.0.0.dev20260512010101"


def test_nightly_falls_back_outside_git_checkout(tmp_path: Path):
    source_root = tmp_path / "source"
    source_root.mkdir()

    version = stamp(
        source_root,
        cadence="nightly",
        release_label="nightly-20260512010101",
        nightly_timestamp="20260512010101",
    )

    assert version == "0.0.0.dev20260512010101"


@pytest.mark.parametrize(
    ("release_label", "expected"),
    [("1.2.3-rc12", "1.2.3rc12"), ("1.2.3-a1", "1.2.3a1"), ("1.2.3-b0", "1.2.3b0")],
)
def test_prerelease_resolves_pep440_version(tmp_path: Path, release_label: str, expected: str):
    version = stamp(tmp_path, cadence="prerelease", release_label=release_label)

    assert version == expected


def test_stable_resolves_release_label(tmp_path: Path):
    version = stamp(tmp_path, cadence="release", release_label="1.0.0")

    assert version == "1.0.0"


@pytest.mark.parametrize("sdk_id", ["../nemo-platform", ".", "..", "bad/id", "bad id"])
def test_unsafe_sdk_id_fails(tmp_path: Path, sdk_id: str):
    with pytest.raises(StampError, match="safe single path segment"):
        stamp(tmp_path, sdk_id=sdk_id)


def test_nemo_platform_plugin_uses_same_resolution_path(tmp_path: Path):
    version = stamp(tmp_path, sdk_id="nemo-platform-plugin", cadence="release", release_label="1.0.0")

    assert version == "1.0.0"


def test_invalid_nightly_timestamp_fails(tmp_path: Path):
    with pytest.raises(StampError, match="nightly timestamp must be YYYYMMDDHHMMSS"):
        stamp(tmp_path, cadence="nightly", release_label="nightly-20260512", nightly_timestamp="20260512")


@pytest.mark.parametrize(
    "release_label",
    [
        "26.05-rc0",
        "1.2.3.4-rc0",
        "01.2.3-rc0",
        "1.02.3-rc0",
        "1.2.03-rc0",
        "1.2.3-alpha.1-rc0",
        "1.2.3+build.1-rc0",
        "1.0.0rc0",
        "1.2.3-alpha1",
        "1.2.3-rc",
        "1.2.3-rc01",
        "1.2.3",
    ],
)
def test_invalid_prerelease_label_fails(tmp_path: Path, release_label: str):
    with pytest.raises(StampError, match="pre-release label must look like"):
        stamp(tmp_path, cadence="prerelease", release_label=release_label)


@pytest.mark.parametrize(
    "release_label",
    ["26.05", "1.2.3.4", "01.2.3", "1.02.3", "1.2.03", "1.2.3-alpha.1", "1.2.3+build.1", "1.0.0-rc0"],
)
def test_invalid_stable_label_fails(tmp_path: Path, release_label: str):
    with pytest.raises(StampError, match="stable release label must be SemVer core"):
        stamp(tmp_path, cadence="release", release_label=release_label)


def test_cli_prints_resolved_version(tmp_path: Path, capsys: pytest.CaptureFixture[str]):
    status = stamp_sdk_version.main(
        [
            "--source-root",
            str(tmp_path),
            "--sdk-id",
            "nemo-platform",
            "--cadence",
            "release",
            "--release-label",
            "1.0.0",
            "--print-version",
        ]
    )

    captured = capsys.readouterr()
    assert status == 0
    assert captured.out == "1.0.0\n"
    assert captured.err == "Resolved sdk:nemo-platform version 1.0.0.\n"
