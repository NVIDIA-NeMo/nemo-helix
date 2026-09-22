#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Callable

from rename_common import (
    BAKE_IMAGE_PATTERN,
    IMAGE_PREFIX,
    LEGACY_ACRONYM_PATTERN,
    LEGACY_PRODUCT_PATTERN,
    content_paths,
    git_file_set,
    read_text,
    repo_root,
)


def print_matches(path: Path, predicate: Callable[[str], object]) -> bool:
    if not path.is_file():
        return False
    text = read_text(path)
    if text is None:
        return False
    found = False
    for line_number, line in enumerate(text.splitlines(), start=1):
        if predicate(line):
            print(f"{path}:{line_number}:{line}")
            found = True
    return found


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify that NeMo Platform rename references are gone.")
    parser.add_argument(
        "--repo-dir",
        type=Path,
        default=Path("."),
        help="repository checkout to verify; defaults to the current working directory",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    os.chdir(repo_root(args.repo_dir))
    failed = False

    paths = content_paths()
    product_matches = [print_matches(path, LEGACY_PRODUCT_PATTERN.search) for path in paths]
    if any(product_matches):
        print("Legacy product names remain in tracked file contents.", file=sys.stderr)
        failed = True

    acronym_matches = [print_matches(path, LEGACY_ACRONYM_PATTERN.search) for path in paths]
    if any(acronym_matches):
        print("Legacy acronym references remain in tracked file contents.", file=sys.stderr)
        failed = True

    for path in git_file_set():
        if not path.exists() and not path.is_symlink():
            continue
        path_string = path.as_posix()
        if LEGACY_PRODUCT_PATTERN.search(path_string) or LEGACY_ACRONYM_PATTERN.search(path_string):
            print(f"Legacy name remains in tracked path: {path}", file=sys.stderr)
            failed = True

    text = read_text(Path("docker-bake.hcl")) or ""
    for image in sorted(set(BAKE_IMAGE_PATTERN.findall(text))):
        if not image.startswith(IMAGE_PREFIX):
            print(f"First-party published image lacks the {IMAGE_PREFIX} prefix: {image}", file=sys.stderr)
            failed = True

    if failed:
        return 1

    print("No legacy product, package, acronym, path, or first-party image names remain.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
