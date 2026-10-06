# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""``NeMoHelix.files``: the filesets transfer helpers plus the generated Files API.

The high-level helpers (``upload``, ``download``, ``list``, ``fsspec``, ...) come
from :mod:`filesets`. The generated sub-resources (``filesets``, ``otlp``) and the
raw/streaming response wrappers come from the generated Files resource.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from filesets.resources import (
    FilesResource as _FilesetsFilesResource,
    AsyncFilesResource as _FilesetsAsyncFilesResource,
)

from .._compat import cached_property
from ..resources.files.files import (
    FilesResource as _GeneratedFilesResource,
    AsyncFilesResource as _GeneratedAsyncFilesResource,
    FilesResourceWithRawResponse,
    AsyncFilesResourceWithRawResponse,
    FilesResourceWithStreamingResponse,
    AsyncFilesResourceWithStreamingResponse,
)
from ..resources.files.filesets import FilesetsResource, AsyncFilesetsResource
from ..resources.files.otlp.otlp import OtlpResource, AsyncOtlpResource

if TYPE_CHECKING:
    # Imported for typing only: _client imports this module.
    from .._client import NeMoHelix, AsyncNeMoHelix

__all__ = ["FilesResource", "AsyncFilesResource"]


class FilesResource(_FilesetsFilesResource):
    def __init__(self, client: NeMoHelix) -> None:
        super().__init__(client)
        self._generated = _GeneratedFilesResource(client)
        # Generated low-level calls, kept for parity with the generated resource.
        self._delete_file = self._generated._delete_file
        self._download_file = self._generated._download_file
        self._list_files = self._generated._list_files
        self._upload_file = self._generated._upload_file

    @cached_property
    def filesets(self) -> FilesetsResource:
        return self._generated.filesets

    @cached_property
    def otlp(self) -> OtlpResource:
        return self._generated.otlp

    @cached_property
    def with_raw_response(self) -> FilesResourceWithRawResponse:
        return self._generated.with_raw_response

    @cached_property
    def with_streaming_response(self) -> FilesResourceWithStreamingResponse:
        return self._generated.with_streaming_response


class AsyncFilesResource(_FilesetsAsyncFilesResource):
    def __init__(self, client: AsyncNeMoHelix) -> None:
        super().__init__(client)
        self._generated = _GeneratedAsyncFilesResource(client)
        # Generated low-level calls, kept for parity with the generated resource.
        self._delete_file = self._generated._delete_file
        self._download_file = self._generated._download_file
        self._list_files = self._generated._list_files
        self._upload_file = self._generated._upload_file

    @cached_property
    def filesets(self) -> AsyncFilesetsResource:
        return self._generated.filesets

    @cached_property
    def otlp(self) -> AsyncOtlpResource:
        return self._generated.otlp

    @cached_property
    def with_raw_response(self) -> AsyncFilesResourceWithRawResponse:
        return self._generated.with_raw_response

    @cached_property
    def with_streaming_response(self) -> AsyncFilesResourceWithStreamingResponse:
        return self._generated.with_streaming_response
