# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared utilities for files tests."""

import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Iterator
from urllib.parse import urlparse

import httpx
from fsspec.spec import AbstractBufferedFile, AbstractFileSystem
from nemo_helix import NeMoHelix
from nemo_helix_plugin.client.adapter import client_from_platform
from nemo_helix_plugin.files.client import FilesClient
from nemo_helix_plugin.files.types import CreateFilesetRequest, FilesetOutput

DEFAULT_WORKSPACE_ID = "default"


def _httpx_path(path: str) -> str:
    """Extract the path component from an httpx:// URL for use with the client."""
    if path.startswith("httpx://"):
        parsed = urlparse(path)
        return parsed.path or "/"
    return path


class HTTPXFileSystem(AbstractFileSystem):
    """An fsspec filesystem that routes requests through an httpx client.

    Supports efficient partial reads via HTTP Range requests, which is important
    for formats like Parquet where only specific byte ranges need to be read.
    """

    protocol = "httpx"
    _fallback_timestamp = datetime.fromtimestamp(0, timezone.utc)

    def __init__(self, client: httpx.Client | None = None, **kwargs):
        super().__init__(**kwargs)
        self.client = client or httpx.Client(follow_redirects=True)

    def _open(
        self,
        path,
        mode="rb",
        block_size="default",
        autocommit=True,
        cache_type="readahead",
        cache_options=None,
        size=None,
        **kwargs,
    ):
        if size is None:
            size = self.info(path).get("size")
        return HTTPXFile(
            self,
            path,
            mode,
            block_size,
            autocommit,
            cache_type,
            cache_options,
            size,
            **kwargs,
        )

    def info(self, path, **kwargs):
        """DuckDB needs the exact file size to find Parquet footers."""
        path = _httpx_path(path)
        resp = self.client.head(path)
        resp.raise_for_status()
        size = int(resp.headers.get("Content-Length", 0))
        return {"name": path, "size": size, "type": "file"}

    def cat_file(self, path, start=None, end=None, **kwargs):
        """Fetch file content with optional byte range."""
        path = _httpx_path(path)
        headers = {}
        if start is not None or end is not None:
            # HTTP Range header: bytes=start-end (inclusive)
            headers["Range"] = f"bytes={start or 0}-{(end - 1) if end else ''}"

        resp = self.client.get(path, headers=headers)
        resp.raise_for_status()
        return resp.content

    def created(self, path):
        self.info(path)
        return self._fallback_timestamp

    def modified(self, path):
        self.info(path)
        return self._fallback_timestamp


class HTTPXFile(AbstractBufferedFile):
    def _fetch_range(self, start, end):
        """Called by fsspec whenever the internal buffer needs more data."""
        return self.fs.cat_file(self.path, start, end)


def test_fileset_name() -> str:
    return f"test-fileset-{uuid.uuid4().hex[:8]}"


@contextmanager
def create_fileset(
    sdk: NeMoHelix,
    name: str | None = None,
    workspace: str = DEFAULT_WORKSPACE_ID,
    **kwargs,
) -> Iterator[FilesetOutput]:
    if name is None:
        name = test_fileset_name()

    files = client_from_platform(sdk, FilesClient)
    fileset = files.create_fileset(
        workspace=workspace,
        body=CreateFilesetRequest(name=name, description="Test fileset", **kwargs),
    ).data()
    yield fileset
    files.delete_fileset(name=name, workspace=workspace)
