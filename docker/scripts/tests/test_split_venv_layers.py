# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import importlib.util
import os
import shutil
import stat
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location("split_venv_layers", Path(__file__).parents[1] / "split-venv-layers.py")
assert SPEC is not None and SPEC.loader is not None
split_venv_layers = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(split_venv_layers)


def _tree(root: Path) -> dict[str, tuple[str, int, bytes | str]]:
    """Every entry under root with its type, permission bits and content or link target."""
    entries: dict[str, tuple[str, int, bytes | str]] = {}
    for dirpath, dirnames, filenames in os.walk(root):
        directory = Path(dirpath)
        for name in dirnames + filenames:
            path = directory / name
            mode = path.lstat().st_mode
            relative = str(path.relative_to(root))
            if stat.S_ISLNK(mode):
                entries[relative] = ("link", 0, os.readlink(path))
            elif stat.S_ISDIR(mode):
                entries[relative] = ("dir", stat.S_IMODE(mode), b"")
            else:
                entries[relative] = ("file", stat.S_IMODE(mode), path.read_bytes())
    return entries


def _copy_layer(source: Path, target: Path) -> None:
    """Merge source into target the way a Dockerfile COPY does: replace files and links, keep directories."""
    for dirpath, dirnames, filenames in os.walk(source):
        directory = Path(dirpath)
        target_dir = target / directory.relative_to(source)
        target_dir.mkdir(exist_ok=True)
        shutil.copystat(directory, target_dir)
        for name in filenames + [name for name in dirnames if (directory / name).is_symlink()]:
            destination = target_dir / name
            if destination.is_symlink() or destination.exists():
                destination.unlink()
            shutil.copy2(directory / name, destination, follow_symlinks=False)


def _write(path: Path, content: str, mode: int = 0o644) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)
    path.chmod(mode)


@pytest.fixture
def venvs(tmp_path: Path) -> tuple[Path, Path]:
    base = tmp_path / "base"
    _write(base / "lib/site-packages/requests/__init__.py", "requests\n")
    _write(base / "lib/site-packages/urllib3-2.7.0.dist-info/METADATA", "old\n")
    _write(base / "lib/site-packages/certifi/cacert.pem", "aaaa\n")
    _write(base / "lib/site-packages/pinned.py", "same size\n")
    _write(base / "bin/tool", "#!/bin/sh\n", 0o755)
    _write(base / "bin/mode-change", "x\n", 0o644)
    (base / "bin/python").symlink_to("/usr/local/bin/python3")
    (base / "lib/site-packages/empty-dir").mkdir()

    venv = tmp_path / "venv"
    shutil.copytree(base, venv, symlinks=True)
    # A workspace member added by the second sync, in a new directory.
    _write(venv / "lib/site-packages/nhx/common/version.py", "VERSION = 1\n")
    _write(venv / "bin/nemo", "#!/bin/sh\n", 0o755)
    # A pinned upgrade: a new dist-info replaces the old one.
    shutil.rmtree(venv / "lib/site-packages/urllib3-2.7.0.dist-info")
    _write(venv / "lib/site-packages/urllib3-2.8.0.dist-info/METADATA", "new\n")
    # Rewritten in place: same size, new content and timestamp.
    _write(venv / "lib/site-packages/certifi/cacert.pem", "bbbb\n")
    # Rewritten with identical content: not part of the overlay.
    _write(venv / "lib/site-packages/pinned.py", "same size\n")
    (venv / "bin/mode-change").chmod(0o755)
    (venv / "bin/python").unlink()
    (venv / "bin/python").symlink_to("/usr/bin/python3")
    (venv / "lib/site-packages/empty-dir").rmdir()
    return base, venv


def test_layers_rebuild_the_finished_venv(tmp_path: Path, venvs: tuple[Path, Path]) -> None:
    base, venv = venvs
    overlay = tmp_path / "overlay"
    removed = tmp_path / "removed.txt"

    split_venv_layers.split(base, venv, overlay, removed)

    # What the image does: copy the base layer, copy the overlay on top, delete the listed paths.
    image = tmp_path / "image"
    shutil.copytree(base, image, symlinks=True)
    _copy_layer(overlay, image)
    listed = removed.read_text().replace(str(venv), str(image))
    image_removed = tmp_path / "image-removed.txt"
    image_removed.write_text(listed)
    split_venv_layers.remove(image_removed)

    assert _tree(image) == _tree(venv)


def test_overlay_holds_only_what_changed(tmp_path: Path, venvs: tuple[Path, Path]) -> None:
    base, venv = venvs
    overlay = tmp_path / "overlay"
    removed = tmp_path / "removed.txt"

    split_venv_layers.split(base, venv, overlay, removed)

    files = {key for key, (kind, _, _) in _tree(overlay).items() if kind != "dir"}
    assert files == {
        "bin/mode-change",
        "bin/nemo",
        "bin/python",
        "lib/site-packages/certifi/cacert.pem",
        "lib/site-packages/nhx/common/version.py",
        "lib/site-packages/urllib3-2.8.0.dist-info/METADATA",
    }
    assert removed.read_text().splitlines() == [
        str(venv / "lib/site-packages/empty-dir"),
        str(venv / "lib/site-packages/urllib3-2.7.0.dist-info"),
    ]


def test_directory_replaced_by_file_is_rejected(tmp_path: Path) -> None:
    base = tmp_path / "base"
    venv = tmp_path / "venv"
    (base / "pkg").mkdir(parents=True)
    _write(base / "pkg/module.py", "x\n")
    _write(venv / "pkg", "now a file\n")

    with pytest.raises(SystemExit, match="is a directory"):
        split_venv_layers.split(base, venv, tmp_path / "overlay", tmp_path / "removed.txt")


def test_remove_fails_when_a_listed_path_is_missing(tmp_path: Path) -> None:
    removed = tmp_path / "removed.txt"
    removed.write_text(f"{tmp_path / 'missing'}\n")

    with pytest.raises(SystemExit, match="listed path is missing"):
        split_venv_layers.remove(removed)
