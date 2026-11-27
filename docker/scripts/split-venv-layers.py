#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Split a virtual environment into a cached third-party layer and a small overlay.

An image copies the third-party venv as one layer, then the overlay on top, then removes the listed paths.
That rebuilds the finished venv exactly while the large third-party layer stays the same blob between builds.

    split  --base <third-party venv> --venv <finished venv> --overlay <dir> --removed <file>
    remove <file>
"""

from __future__ import annotations

import argparse
import filecmp
import os
import shutil
import stat
import sys
from pathlib import Path


def _same_file(base: Path, base_stat: os.stat_result, path: Path, path_stat: os.stat_result) -> bool:
    if stat.S_IFMT(base_stat.st_mode) != stat.S_IFMT(path_stat.st_mode):
        return False
    if stat.S_ISLNK(path_stat.st_mode):
        return os.readlink(base) == os.readlink(path)
    if stat.S_IMODE(base_stat.st_mode) != stat.S_IMODE(path_stat.st_mode):
        return False
    if base_stat.st_size != path_stat.st_size:
        return False
    if base_stat.st_mtime_ns == path_stat.st_mtime_ns:
        return True
    return filecmp.cmp(base, path, shallow=False)


def _lstat(path: Path) -> os.stat_result | None:
    try:
        return path.lstat()
    except FileNotFoundError:
        return None


def _ensure_dir(overlay: Path, venv: Path, relative: Path) -> None:
    """Create overlay/relative and its parents with the modes the finished venv has."""
    current = Path()
    for part in relative.parts:
        current /= part
        target = overlay / current
        if not target.exists():
            target.mkdir()
            shutil.copystat(venv / current, target, follow_symlinks=False)


def split(base: Path, venv: Path, overlay: Path, removed: Path) -> tuple[int, int, int]:
    """Stage what `venv` adds to or changes in `base` under `overlay`, and list what it removes."""
    if overlay.exists():
        raise SystemExit(f"{overlay} already exists")
    overlay.mkdir(parents=True)
    copied = 0
    copied_bytes = 0
    for dirpath, dirnames, filenames in os.walk(venv):
        directory = Path(dirpath)
        relative_dir = directory.relative_to(venv)
        base_dir_stat = _lstat(base / relative_dir)
        if base_dir_stat is not None and not stat.S_ISDIR(base_dir_stat.st_mode):
            raise SystemExit(f"{relative_dir} is a directory in {venv} but not in {base}")
        if base_dir_stat is None or stat.S_IMODE(base_dir_stat.st_mode) != stat.S_IMODE(directory.lstat().st_mode):
            _ensure_dir(overlay, venv, relative_dir)
        entries = list(filenames)
        # os.walk lists symlinks to directories as directories; treat them as files.
        for name in list(dirnames):
            if (directory / name).is_symlink():
                dirnames.remove(name)
                entries.append(name)
        for name in sorted(entries):
            path = directory / name
            relative = relative_dir / name
            path_stat = path.lstat()
            base_stat = _lstat(base / relative)
            if base_stat is not None and stat.S_ISDIR(base_stat.st_mode):
                raise SystemExit(f"{relative} is a directory in {base} but not in {venv}")
            if base_stat is not None and _same_file(base / relative, base_stat, path, path_stat):
                continue
            _ensure_dir(overlay, venv, relative_dir)
            if stat.S_ISLNK(path_stat.st_mode):
                os.symlink(os.readlink(path), overlay / relative)
            else:
                shutil.copy2(path, overlay / relative, follow_symlinks=False)
                copied_bytes += path_stat.st_size
            copied += 1

    removed_paths: list[str] = []
    for dirpath, dirnames, filenames in os.walk(base):
        directory = Path(dirpath)
        relative_dir = directory.relative_to(base)
        for name in sorted(dirnames):
            if (directory / name).is_symlink():
                filenames.append(name)
        dirnames[:] = [name for name in dirnames if not (directory / name).is_symlink()]
        for name in list(dirnames):
            if _lstat(venv / relative_dir / name) is None:
                removed_paths.append(str(venv / relative_dir / name))
                dirnames.remove(name)
        for name in sorted(filenames):
            if _lstat(venv / relative_dir / name) is None:
                removed_paths.append(str(venv / relative_dir / name))
    for path in removed_paths:
        if "\n" in path:
            raise SystemExit(f"cannot list a path containing a newline: {path!r}")
    removed.write_text("".join(f"{path}\n" for path in sorted(removed_paths)), encoding="utf-8")
    return copied, copied_bytes, len(removed_paths)


def remove(removed: Path) -> int:
    """Delete every path the split listed. A listed path that is missing means the layers don't match."""
    count = 0
    for line in removed.read_text(encoding="utf-8").splitlines():
        if not line:
            continue
        path = Path(line)
        path_stat = _lstat(path)
        if path_stat is None:
            raise SystemExit(f"listed path is missing: {path}")
        if stat.S_ISDIR(path_stat.st_mode):
            shutil.rmtree(path)
        else:
            path.unlink()
        count += 1
    return count


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    split_parser = commands.add_parser("split")
    split_parser.add_argument("--base", type=Path, required=True)
    split_parser.add_argument("--venv", type=Path, required=True)
    split_parser.add_argument("--overlay", type=Path, required=True)
    split_parser.add_argument("--removed", type=Path, required=True)
    remove_parser = commands.add_parser("remove")
    remove_parser.add_argument("removed", type=Path)
    args = parser.parse_args(argv)
    if args.command == "split":
        copied, copied_bytes, removed = split(args.base, args.venv, args.overlay, args.removed)
        print(f"venv overlay: {copied} files, {copied_bytes / 1e6:.1f} MB; {removed} removed paths", file=sys.stderr)
    else:
        print(f"removed {remove(args.removed)} paths", file=sys.stderr)


if __name__ == "__main__":
    main()
