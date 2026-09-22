#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path

from rename_common import (
    ACRONYM_REPLACEMENT_RULES,
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


def inventory(include_globs: tuple[str, ...], exclude_globs: tuple[str, ...]) -> None:
    paths = content_paths(include_globs, exclude_globs)
    print("Legacy content categories:")
    for old, new in PRODUCT_REPLACEMENTS:
        count = 0
        for path in paths:
            text = read_text(path)
            if text is None:
                continue
            count += sum(1 for line in text.splitlines() if old in line)
        print(f"  {old:<24} -> {new:<24} {count:8d} matching lines")
    for old, pattern, new in ACRONYM_REPLACEMENT_RULES:
        count = 0
        for path in paths:
            text = read_text(path)
            if text is None:
                continue
            count += sum(1 for line in text.splitlines() for _ in pattern.finditer(line))
        print(f"  {old:<24} -> {new:<24} {count:8d} matching lines")

    print()
    print("Legacy tracked paths:")
    for path in tracked_paths(include_globs, exclude_globs):
        renamed = renamed_path(path)
        if renamed != path:
            print(f"  {path} -> {renamed}")

    print()
    print("First-party published image renames:")
    if Path("docker-bake.hcl") not in paths:
        return
    text = read_text(Path("docker-bake.hcl")) or ""
    for image in sorted(set(BAKE_IMAGE_PATTERN.findall(text))):
        renamed_image = replace_legacy_names(image)
        if not renamed_image.startswith(IMAGE_PREFIX):
            renamed_image = f"{IMAGE_PREFIX}{renamed_image}"
        if renamed_image != image:
            print(f"  {image} -> {renamed_image}")


def apply_content_replacements(include_globs: tuple[str, ...], exclude_globs: tuple[str, ...]) -> None:
    for path in content_paths(include_globs, exclude_globs):
        if path.is_symlink() or not path.is_file():
            continue
        text = read_text(path)
        if text is None:
            continue
        updated = replace_text(text)
        if updated != text:
            path.write_bytes(updated.encode("utf-8"))


def apply_path_renames(include_globs: tuple[str, ...], exclude_globs: tuple[str, ...]) -> None:
    for path in git_file_set(include_globs, exclude_globs):
        if not path.exists() and not path.is_symlink():
            continue
        destination = renamed_path(path)
        if destination == path:
            continue
        if destination.exists() or destination.is_symlink():
            raise RuntimeError(f"Cannot rename {path}: destination already exists: {destination}")
        destination.parent.mkdir(parents=True, exist_ok=True)
        path.rename(destination)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rename NeMo Platform references to NeMo Helix.")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true", help="print planned changes without editing files")
    mode.add_argument("--continue", dest="resume", action="store_true", help="resume after a partial rename")
    parser.add_argument(
        "--repo-dir",
        type=Path,
        default=Path("."),
        help="repository checkout to modify; defaults to the current working directory",
    )
    parser.add_argument(
        "--include-glob",
        action="append",
        default=[],
        metavar="PATTERN",
        help="only modify repo-relative paths matching this glob; may be repeated",
    )
    parser.add_argument(
        "--exclude-glob",
        action="append",
        default=[],
        metavar="PATTERN",
        help="skip repo-relative paths matching this glob; may be repeated",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    os.chdir(repo_root(args.repo_dir))
    include_globs = tuple(args.include_glob)
    exclude_globs = tuple(args.exclude_glob)
    if args.dry_run:
        inventory(include_globs, exclude_globs)
        return 0
    if not args.resume and run_git("status", "--short").stdout:
        print("The worktree must be clean before running the rename.", file=sys.stderr)
        return 1

    try:
        apply_content_replacements(include_globs, exclude_globs)
        apply_path_renames(include_globs, exclude_globs)
    except RuntimeError as error:
        print(error, file=sys.stderr)
        return 1

    print("Rename complete. Run tools/rename/verify-nemo-helix-rename.sh before committing.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
