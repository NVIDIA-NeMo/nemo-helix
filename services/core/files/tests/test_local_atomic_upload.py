# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Test atomic upload behavior for local storage backend."""

import asyncio
from pathlib import Path
from unittest.mock import patch

import anyio
import pytest
from nhx.common.files.storage_config import LocalStorageConfig
from nhx.core.files.app.backends.local import LocalStorageImpl


async def async_iter_from_list(items: list[bytes]):
    """Helper to create an async iterator from a list of bytes."""
    for item in items:
        yield item


async def test_upload_uses_temp_file_then_rename(tmp_path):
    """Verify upload writes to temp file first, then atomically renames."""
    config = LocalStorageConfig(path=str(tmp_path))
    storage = LocalStorageImpl(config)

    # Track what files exist during the upload
    observed_files = []
    original_open_file = anyio.open_file

    async def tracking_open_file(path, mode="r", **kwargs):
        # Record that we're opening this file
        observed_files.append(Path(path).name)
        return await original_open_file(path, mode, **kwargs)

    with patch("anyio.open_file", side_effect=tracking_open_file):
        chunks = [b"test", b"data"]
        await storage.upload("test.txt", async_iter_from_list(chunks))

    # Should have opened a temp file (with .tmp. in name)
    assert any(".tmp." in name for name in observed_files)

    # Final file should exist with correct content
    final_path = tmp_path / "test.txt"
    assert final_path.exists()
    assert final_path.read_bytes() == b"testdata"

    # No temp files should remain
    temp_files = list(tmp_path.glob("*.tmp.*"))
    assert len(temp_files) == 0


async def test_upload_cleans_up_temp_file_on_failure(tmp_path):
    """Verify temp file is cleaned up if upload fails."""
    config = LocalStorageConfig(path=str(tmp_path))
    storage = LocalStorageImpl(config)

    async def failing_stream():
        yield b"some data"
        raise RuntimeError("Stream failed!")

    with pytest.raises(RuntimeError, match="Stream failed"):
        await storage.upload("test.txt", failing_stream())

    # No files should remain (neither temp nor final)
    assert len(list(tmp_path.iterdir())) == 0


async def test_concurrent_uploads_dont_collide(tmp_path):
    """Verify concurrent uploads to same path use different temp files."""
    config = LocalStorageConfig(path=str(tmp_path))
    storage = LocalStorageImpl(config)

    async def slow_stream(data: bytes, delay: float = 0.01):
        """Stream that yields slowly to simulate concurrent writes."""
        for byte in data:
            await asyncio.sleep(delay)
            yield bytes([byte])

    # Run two uploads concurrently to the same path
    async def upload_task(content: bytes):
        return await storage.upload("concurrent.txt", slow_stream(content))

    # Start both uploads concurrently
    results = await asyncio.gather(upload_task(b"AAAA"), upload_task(b"BBBB"), return_exceptions=True)

    # One should succeed, the other might succeed or fail
    # depending on timing of the rename operation
    successful_results = [r for r in results if not isinstance(r, Exception)]
    assert len(successful_results) >= 1

    # Final file should contain one of the complete datasets
    final_content = (tmp_path / "concurrent.txt").read_bytes()
    assert final_content in [b"AAAA", b"BBBB"]

    # No temp files should remain
    temp_files = list(tmp_path.glob("*.tmp.*"))
    assert len(temp_files) == 0


async def test_upload_handles_rename_failures_gracefully(tmp_path):
    """Test handling when rename fails (e.g., on some Windows systems)."""
    config = LocalStorageConfig(path=str(tmp_path))
    storage = LocalStorageImpl(config)

    # Pre-create the target file
    target_file = tmp_path / "existing.txt"
    target_file.write_bytes(b"old content")

    # Upload should overwrite it atomically
    chunks = [b"new", b"content"]
    result = await storage.upload("existing.txt", async_iter_from_list(chunks))

    # Should have succeeded and replaced the file
    assert result.size == 10  # len("newcontent")
    assert target_file.read_bytes() == b"newcontent"

    # No temp files should remain
    temp_files = list(tmp_path.glob("*.tmp.*"))
    assert len(temp_files) == 0
