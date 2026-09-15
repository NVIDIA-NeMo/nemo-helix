# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Line-delimited JSON reader. No declared schema; the row count is exact only on a full read."""

from __future__ import annotations

import json
from collections.abc import Iterator
from typing import BinaryIO

from nemo_profiler_plugin.dataset.readers.base import FilePreview, ReadResult, register_reader
from nemo_profiler_plugin.source import FileEntry, FileSource

# Rows per batch, matching the parquet reader so the caller's working set does not depend on the
# format being folded.
_BATCH_ROWS = 1024

# Signatures of the formats a `.jsonl` most often turns out to be. Only *binary* ones are listed,
# which is the discipline: none of these bytes can begin a text file, so the check cannot produce a
# false positive. A file that is text but not line-delimited objects -- a pretty-printed array, a
# file of bare scalars -- is left to `_records`, which reports "skipped N line(s)".
_BINARY_MAGIC: tuple[tuple[bytes, str], ...] = (
    (b"PAR1", "a parquet file"),
    (b"\x1f\x8b", "a gzip archive"),
    (b"PK\x03\x04", "a zip archive"),
    (b"ARROW1", "an arrow IPC file"),
)

# Enough for the longest signature above. Read once, from a stream the caller is opening anyway.
_SNIFF_BYTES = 16


def _check_magic(stream: BinaryIO) -> None:
    """Reject a file whose leading bytes say it is some other format entirely.

    Line-delimited JSON has no signature of its own, so this can only say what the file is *not*.
    That is still the difference between "begins with the magic bytes of a gzip archive" and forty
    thousand lines of ``Expecting value``, which reads like corrupt data.

    Called from every entry point rather than :meth:`JsonlReader.peek` alone, because the pipeline
    discards what a peek raises and the read path is where this reaches a ``FileError``.
    """
    head = stream.read(_SNIFF_BYTES)
    stream.seek(0)
    for magic, description in _BINARY_MAGIC:
        if head.startswith(magic):
            raise ValueError(f"not line-delimited JSON: the file begins with the magic bytes of {description}")


# The longest single line the reader will hold in memory. A `.jsonl` whose rows are rows never comes
# near this; what does is the malformed input this module already anticipates -- a pretty-printed
# JSON array saved under the wrong extension, which is one line as long as the file. Reading that
# with `for line in stream` allocated the whole file as one bytes object, then another for `.strip()`,
# then the parsed tree, before discovering it was not a row. `row_cap` cannot help, because the cap
# is checked after the line has been read.
_MAX_LINE_BYTES = 64 * 1024 * 1024


def _lines(stream: BinaryIO) -> Iterator[bytes | None]:
    """Each line of the stream, or None for one too long to hold.

    An over-long line is drained rather than buffered, so the file keeps its line numbering and its
    remaining rows still parse.
    """
    while True:
        # One byte past the bound, so that "stopped at the limit" and "the line is longer than the
        # limit" are different lengths. Reading exactly the bound conflated them, and a final line of
        # exactly `_MAX_LINE_BYTES` -- read whole, nothing left of it -- was discarded as over-long.
        chunk = stream.readline(_MAX_LINE_BYTES + 1)
        if not chunk:
            return
        if len(chunk) > _MAX_LINE_BYTES and not chunk.endswith(b"\n"):
            while True:
                more = stream.readline(_MAX_LINE_BYTES)
                if not more or more.endswith(b"\n"):
                    break
            yield None
            continue
        yield chunk


def _records(stream) -> Iterator[tuple[dict | None, str | None]]:
    """Each line of the stream as either a record or a reason it was not one.

    Three outcomes, not two. A record; a failure, with the reason; or ``(None, None)`` for a line
    that parsed but is not a row -- a stray scalar or array. Blank lines are separator and are not
    reported at all.

    Shared by :meth:`JsonlReader.read` and :meth:`JsonlReader.batches` so the two cannot disagree
    on what counts as a row.
    """
    for line_number, raw_line in enumerate(_lines(stream), start=1):
        if raw_line is None:
            yield None, f"line {line_number}: longer than {_MAX_LINE_BYTES} bytes; not read"
            continue
        stripped = raw_line.strip()
        if not stripped:  # tolerate blank lines between records
            continue
        try:
            record = json.loads(stripped)
        except ValueError as exc:
            # A truncated or corrupt line costs that line, never the file. Dropping the whole file
            # would erase its row count and every column it was the only witness for.
            yield None, f"line {line_number}: {exc}"
            continue
        # A row is a column map. A stray scalar or array is not one, and is not a failure either --
        # it is simply not a row of this dataset, so it costs nothing and the count stays exact.
        yield (record, None) if isinstance(record, dict) else (None, None)


class JsonlReader:
    file_format = "jsonl"

    def peek(self, source: FileSource, entry: FileEntry) -> FilePreview:
        """Nothing but a format check. A line-delimited file declares no schema and carries no row
        count, which is why a partition holding one cannot be folded without reading it first."""
        with source.open(entry.path) as stream:
            _check_magic(stream)
        return FilePreview()

    def batches(
        self,
        source: FileSource,
        entry: FileEntry,
        *,
        row_cap: int | None = None,
        errors: list[str] | None = None,
    ) -> Iterator[list[dict]]:
        """Rows in chunks, reporting any line it could not read into ``errors``.

        A corrupt line costs that line, never the file: dropping the file would erase its row count
        and every column it alone witnessed. The caller still has to be told, or a partially parsed
        file folds silently and looks complete.
        """
        rows: list[dict] = []
        scanned = 0
        unusable = 0
        not_rows = 0
        first_failure: str | None = None
        with source.open(entry.path) as stream:
            # Below the format check, not above it, as the parquet reader has it. Returning first
            # made a zero cap skip validation altogether: a gzip archive saved as `.jsonl` was
            # reported at every budget except that one, where it passed silently as a readable
            # file holding no rows. A cap says how much to read, not whether to look.
            _check_magic(stream)
            if row_cap == 0:
                return
            for record, failure in _records(stream):
                if record is None:
                    if failure is None:
                        not_rows += 1
                        continue
                    unusable += 1
                    if first_failure is None:
                        first_failure = failure
                    continue
                rows.append(record)
                scanned += 1
                if len(rows) >= _BATCH_ROWS:
                    yield rows
                    rows = []
                if row_cap is not None and scanned >= row_cap:
                    break
        if rows:
            yield rows
        if errors is not None:
            if unusable:
                errors.append(f"skipped {unusable} line(s); first at {first_failure}")
            # Lines that were not rows cost nothing while *some* line was one. When none was, this is
            # not an empty dataset -- it is a file this reader could not use, and saying nothing would
            # make it indistinguishable from a dataset that really is empty. A pretty-printed JSON
            # array saved as `.jsonl` is exactly one such file, on one line.
            elif not_rows and not scanned:
                errors.append(f"no JSON object rows in {not_rows} line(s); this may not be line-delimited JSON")

    def read(self, source: FileSource, entry: FileEntry, *, row_cap: int | None = None) -> ReadResult:
        rows: list[dict] = []
        unusable = 0
        not_rows = 0
        first_failure: str | None = None
        hit_cap = False
        with source.open(entry.path) as stream:
            # Tested before the loop rather than after each row: `len(rows) >= row_cap` only fires
            # once a row has been appended, so a zero cap still read and folded one row per file --
            # the one value `_per_file_cap` exists to define. Below the format check, though, since
            # a cap says how much to read and not whether to look.
            _check_magic(stream)
            if row_cap == 0:
                return ReadResult(rows=[], examples_scanned=0, num_rows=None, arrow_schema=None)
            for record, failure in _records(stream):
                # Branch on the record, not the failure: the two are exclusive, and this narrows the
                # type without an ignore comment standing in for the reasoning.
                if record is None:
                    if failure is None:
                        not_rows += 1
                        continue
                    unusable += 1
                    if first_failure is None:
                        first_failure = failure
                    continue
                rows.append(record)
                if row_cap is not None and len(rows) >= row_cap:
                    hit_cap = True
                    break

        # `num_rows` counts records the reader could parse. A line of valid JSON that is not a row (a
        # stray scalar) is not a row of this dataset, so it leaves the count exact and sets no error;
        # an unusable line is data we failed to read, so it is reported and the pipeline uses
        # `error` to decide the file was not exhaustively scanned. A cap costs the exact count only
        # when it actually stopped the read -- a file smaller than the cap was read to EOF and keeps
        # an exact count, which lets a capped profile of a small dataset stay exhaustive.
        num_rows = None if hit_cap else len(rows)
        error = f"skipped {unusable} line(s); first at {first_failure}" if unusable else None
        if error is None and not_rows and not rows:
            error = f"no JSON object rows in {not_rows} line(s); this may not be line-delimited JSON"
        return ReadResult(rows=rows, examples_scanned=len(rows), num_rows=num_rows, arrow_schema=None, error=error)


register_reader(JsonlReader())
