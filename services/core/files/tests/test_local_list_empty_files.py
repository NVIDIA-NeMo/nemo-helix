# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Listing must include empty files."""

from collections.abc import AsyncIterator
from pathlib import Path

from nhx.core.files.app.backends.local import LocalStorageConfig, LocalStorageImpl


async def test_list_files_includes_empty_files(tmp_path: Path) -> None:
    """A zero-byte file is listed with size 0, next to files that have content."""
    storage = LocalStorageImpl(LocalStorageConfig(path=str(tmp_path)))

    async def empty() -> AsyncIterator[bytes]:
        for chunk in ():
            yield chunk

    async def body() -> AsyncIterator[bytes]:
        yield b"data"

    await storage.upload("empty.bin", empty())
    await storage.upload("full.bin", body())

    files = await storage.list_files()
    assert {file.path: file.size for file in files} == {"empty.bin": 0, "full.bin": 4}
