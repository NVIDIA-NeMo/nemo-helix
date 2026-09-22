#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os
import re
import sys
from pathlib import Path

from rename_common import (
    ACRONYM_PATTERN,
    ACRONYM_REPLACEMENTS,
    BAKE_IMAGE_PATTERN,
    IMAGE_PATTERN,
    IMAGE_PREFIX,
    PRODUCT_REPLACEMENTS,
    content_paths,
    git_file_set,
    read_text,
    renamed_path,
    replace_legacy_names,
    repo_root,
    run_git,
    tracked_paths,
)


def rename_image(match: re.Match[str]) -> str:
    return f"{IMAGE_PREFIX}{match.group(1)}"


def replace_text(text: str) -> str:
    return IMAGE_PATTERN.sub(rename_image, replace_legacy_names(text))


def inventory() -> None:
    paths = content_paths()
    print("Legacy content categories:")
    for old, new in PRODUCT_REPLACEMENTS:
        count = 0
        for path in paths:
            text = read_text(path)
            if text is None:
                continue
            count += sum(1 for line in text.splitlines() if old in line)
        print(f"  {old:<24} -> {new:<24} {count:8d} matching lines")
    for old, new in ACRONYM_REPLACEMENTS.items():
        count = 0
        for path in paths:
            text = read_text(path)
            if text is None:
                continue
            count += sum(match == old for line in text.splitlines() for match in ACRONYM_PATTERN.findall(line))
        print(f"  {old:<24} -> {new:<24} {count:8d} matching lines")

    print()
    print("Legacy tracked paths:")
    for path in tracked_paths():
        renamed = renamed_path(path)
        if renamed != path:
            print(f"  {path} -> {renamed}")

    print()
    print("First-party published image renames:")
    text = read_text(Path("docker-bake.hcl")) or ""
    for image in sorted(set(BAKE_IMAGE_PATTERN.findall(text))):
        renamed_image = replace_legacy_names(image)
        if not renamed_image.startswith(IMAGE_PREFIX):
            renamed_image = f"{IMAGE_PREFIX}{renamed_image}"
        if renamed_image != image:
            print(f"  {image} -> {renamed_image}")


def apply_content_replacements() -> None:
    for path in content_paths():
        if path.is_symlink() or not path.exists():
            continue
        text = read_text(path)
        if text is None:
            continue
        updated = replace_text(text)
        if updated != text:
            path.write_bytes(updated.encode("utf-8"))


def apply_path_renames() -> None:
    for path in git_file_set():
        if not path.exists() and not path.is_symlink():
            continue
        destination = renamed_path(path)
        if destination == path:
            continue
        if destination.exists() or destination.is_symlink():
            raise RuntimeError(f"Cannot rename {path}: destination already exists: {destination}")
        destination.parent.mkdir(parents=True, exist_ok=True)
        path.rename(destination)


def usage() -> None:
    print(f"Usage: {sys.argv[0]} [--dry-run|--continue]")


def main() -> int:
    os.chdir(repo_root())
    if len(sys.argv) > 2:
        usage()
        return 2

    mode = sys.argv[1] if len(sys.argv) == 2 else ""
    if mode == "--dry-run":
        inventory()
        return 0
    if mode and mode != "--continue":
        usage()
        return 2
    if mode != "--continue" and run_git("status", "--short").stdout:
        print("The worktree must be clean before running the rename.", file=sys.stderr)
        return 1

    try:
        apply_content_replacements()
        apply_path_renames()
    except RuntimeError as error:
        print(error, file=sys.stderr)
        return 1

    print("Rename complete. Run tools/rename/verify-nemo-helix-rename.sh before committing.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
