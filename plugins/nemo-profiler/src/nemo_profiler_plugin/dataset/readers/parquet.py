# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Parquet reader — the footer gives an exact row count and the declared schema for free."""

from __future__ import annotations

import io
from collections.abc import Iterator
from typing import BinaryIO

import pyarrow.parquet as pq
from nemo_profiler_plugin.dataset.readers.base import FilePreview, ReadResult, register_reader
from nemo_profiler_plugin.source import FileEntry, FileSource

# Rows per batch. Small enough that the working set is independent of the dataset, large enough
# that per-batch overhead stays invisible.
_BATCH_ROWS = 1024

# Every parquet file opens and closes with these four bytes; the trailing copy is what marks the
# footer. https://parquet.apache.org/docs/file-format/
_MAGIC = b"PAR1"


def _check_magic(stream: BinaryIO) -> None:
    """Confirm the stream is a parquet file before pyarrow tries to read one.

    Format is decided by extension, so anything named ``.parquet`` reaches this reader: a truncated
    upload, a file that was never parquet, an HTML error page saved under the wrong name. Left to
    pyarrow all three come back as a decode failure from inside the footer.

    Checking both markers separates the two answers worth having: a missing leading marker means this
    was never parquet, a missing trailing one means it was and the bytes stop early.

    Called from every entry point, because the pipeline discards peek failures and the read path is
    where this reaches a ``FileError``.
    """
    size = stream.seek(0, io.SEEK_END)
    if size < len(_MAGIC) * 2:
        raise ValueError(f"not a parquet file: {size} bytes cannot hold both PAR1 markers")
    stream.seek(0)
    if stream.read(len(_MAGIC)) != _MAGIC:
        raise ValueError("not a parquet file: no PAR1 marker at the start")
    stream.seek(-len(_MAGIC), io.SEEK_END)
    if stream.read(len(_MAGIC)) != _MAGIC:
        raise ValueError("truncated parquet file: no PAR1 marker at the end")
    stream.seek(0)


class ParquetReader:
    file_format = "parquet"

    def peek(self, source: FileSource, entry: FileEntry) -> FilePreview:
        """Schema and exact row count from the footer. Reads no rows, so a partition's shape costs
        one seek per file."""
        with source.open(entry.path) as stream:
            _check_magic(stream)
            parquet_file = pq.ParquetFile(stream)
            return FilePreview(arrow_schema=parquet_file.schema_arrow, num_rows=parquet_file.metadata.num_rows)

    def batches(
        self,
        source: FileSource,
        entry: FileEntry,
        *,
        row_cap: int | None = None,
        errors: list[str] | None = None,
    ) -> Iterator[list[dict]]:
        """Rows in chunks, so the caller can fold them and let each chunk go.

        ``errors`` is never appended to: parquet either decodes a batch or raises, so there is no
        partial understanding to report.
        """
        scanned = 0
        with source.open(entry.path) as stream:
            _check_magic(stream)
            parquet_file = pq.ParquetFile(stream)
            if row_cap == 0:
                return
            for batch in parquet_file.iter_batches(
                batch_size=_BATCH_ROWS if row_cap is None else min(row_cap, _BATCH_ROWS)
            ):
                rows = batch.to_pylist()
                if row_cap is not None and scanned + len(rows) > row_cap:
                    rows = rows[: row_cap - scanned]
                scanned += len(rows)
                if rows:
                    yield rows
                if row_cap is not None and scanned >= row_cap:
                    return

    def read(self, source: FileSource, entry: FileEntry, *, row_cap: int | None = None) -> ReadResult:
        with source.open(entry.path) as stream:
            _check_magic(stream)
            parquet_file = pq.ParquetFile(stream)
            num_rows = parquet_file.metadata.num_rows
            arrow_schema = parquet_file.schema_arrow
            if row_cap == 0:
                return ReadResult(rows=[], examples_scanned=0, num_rows=num_rows, arrow_schema=arrow_schema)

            rows: list[dict] = []
            # Same ceiling `batches` applies. Passing `row_cap` straight through asked pyarrow for a
            # single RecordBatch that size, so a large cap allocated it in arrow before any of it was
            # converted. This bounds that transient allocation only: `read` returns the rows, so
            # `rows` still grows to `row_cap` by contract. `batches` is the method whose working set
            # is independent of the cap.
            for batch in parquet_file.iter_batches(
                batch_size=_BATCH_ROWS if row_cap is None else min(row_cap, _BATCH_ROWS)
            ):
                rows.extend(batch.to_pylist())
                if row_cap is not None and len(rows) >= row_cap:
                    break

        if row_cap is not None:
            rows = rows[:row_cap]
        return ReadResult(rows=rows, examples_scanned=len(rows), num_rows=num_rows, arrow_schema=arrow_schema)


register_reader(ParquetReader())
