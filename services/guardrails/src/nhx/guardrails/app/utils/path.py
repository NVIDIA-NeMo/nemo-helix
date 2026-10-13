# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import shutil
from pathlib import Path


def copy_folder(src_path: Path, dst_path: Path):
    """Copy the contents of a folder to another folder.

    Args:
        src_path (Path): The source folder to copy from.
        dst_path (Path): The destination folder to copy to.

    """
    # Ensure the destination directory exists
    dst_path.mkdir(parents=True, exist_ok=True)

    # Iterate over all items in source directory recursively
    for item in src_path.rglob("*"):
        # Calculate the destination path for each item
        dest = dst_path / item.relative_to(src_path)

        # Copy files and create directories as needed
        if item.is_dir():
            dest.mkdir(exist_ok=True)
        else:
            dest.parent.mkdir(parents=True, exist_ok=True)  # Ensure parent directory exists
            if item.resolve() != dest.resolve():
                shutil.copy2(item, dest)
