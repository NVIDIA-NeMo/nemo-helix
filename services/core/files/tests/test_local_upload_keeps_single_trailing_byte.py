# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from nhx.core.files.app.backends.local import LocalStorageConfig, LocalStorageImpl


async def test_upload_keeps_single_trailing_byte(tmp_path):
    storage = LocalStorageImpl(LocalStorageConfig(path=str(tmp_path)))

    async def one_byte():
        yield b"Z"

    info = await storage.upload("single/one.bin", one_byte())
    assert info.size == 1

    chunks = [chunk async for chunk in await storage.download("single/one.bin", None)]
    assert b"".join(chunks) == b"Z"
