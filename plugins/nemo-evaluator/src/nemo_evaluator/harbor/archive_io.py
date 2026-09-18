# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Bounded, authenticated streaming reads shared by publication and materialization."""

import hashlib
from typing import BinaryIO

from filesets import parse_fileset_ref
from nemo_evaluator.harbor.archive import CHUNK_BYTES
from nemo_platform_plugin.files.client import AsyncFilesClient, FilesClient


class _CheckedWriter:
    def __init__(self, output: BinaryIO, limit: int) -> None:
        self.output = output
        self.limit = limit
        self.size = 0
        self.digest = hashlib.sha256()

    def write(self, chunk: bytes) -> None:
        self.size += len(chunk)
        if self.size > self.limit:
            raise ValueError("Downloaded object exceeds byte limit")
        self.digest.update(chunk)
        self.output.write(chunk)

    def verify(self, expected_size: int | None, expected_digest: str | None) -> None:
        if expected_size is not None and self.size != expected_size:
            raise ValueError("Downloaded object size mismatch")
        if expected_digest is not None and self.digest.hexdigest() != expected_digest:
            raise ValueError("Downloaded object checksum mismatch")


def download_verified(
    client: FilesClient,
    ref: str,
    output: BinaryIO,
    *,
    limit: int,
    expected_size: int | None = None,
    expected_digest: str,
) -> None:
    workspace, name, path = parse_fileset_ref(ref, workspace_fallback=None)
    writer = _CheckedWriter(output, limit)
    response = client.download_file(workspace=workspace, name=name, path=path)
    with response.stream(chunk_size=CHUNK_BYTES) as chunks:
        for chunk in chunks:
            writer.write(chunk)
    writer.verify(expected_size, expected_digest)


async def download_verified_async(
    client: AsyncFilesClient,
    ref: str,
    output: BinaryIO,
    *,
    limit: int,
    expected_size: int | None = None,
    expected_digest: str,
) -> None:
    workspace, name, path = parse_fileset_ref(ref, workspace_fallback=None)
    writer = _CheckedWriter(output, limit)
    response = await client.download_file(workspace=workspace, name=name, path=path)
    async with response.stream(chunk_size=CHUNK_BYTES) as chunks:
        async for chunk in chunks:
            writer.write(chunk)
    writer.verify(expected_size, expected_digest)
