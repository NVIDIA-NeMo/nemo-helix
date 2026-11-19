# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""A single trailing byte must survive upload: the final buffer flush must not drop it."""

from nhx.core.files.app.backends.local import LocalStorageConfig, LocalStorageImpl


async def test_single_byte_upload_roundtrip(tmp_path):
    storage = LocalStorageImpl(LocalStorageConfig(path=str(tmp_path)))

    async def body():
        yield b"Z"

    info = await storage.upload("tiny.bin", body())
    assert info.size == 1

    chunks = [chunk async for chunk in await storage.download("tiny.bin", None)]
    assert b"".join(chunks) == b"Z"
