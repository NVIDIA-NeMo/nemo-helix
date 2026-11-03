# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The checks of a customization dry run (create with ``?dry_run=true``).

Besides the execution-profile check every dry run makes, a customization dry run reads
the first rows of each file training would read, or every row with ``?dry_run_full=true``,
and checks them with the backend's own validator. It reads byte ranges, or for every row a
stream, from the Files service, so nothing is downloaded to job storage and nothing is created.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Callable, Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path

from nemo_helix_plugin.client.client import AsyncNemoClient
from nemo_helix_plugin.files.client import AsyncFilesClient
from nemo_helix_plugin.jobs.types import JobDryRunCheck
from nhx.customization_common.schemas.file_io import FileSetRef

ROWS_TO_CHECK = 10
_CHUNK_BYTES = 64 * 1024
# The most read from a file. A first row longer than this fails the check rather than being read in full.
_MAX_BYTES = 4 * 1024 * 1024


@dataclass(frozen=True)
class DatasetRows:
    """A dataset a job reads, and how training picks and checks its files."""

    #: What the check reports, e.g. ``"training dataset"``.
    name: str
    #: The fileset reference from the job spec.
    fileset: str
    #: Training's own file picker: relative paths in, the files training reads out.
    select_files: Callable[[list[str]], list[str]]
    #: The backend's row validator: ``(lines, source)``; raises on a bad row.
    validate_lines: Callable[[Iterable[str], str], object]


async def check_dataset_rows(
    datasets: list[DatasetRows], workspace: str, async_sdk: AsyncNemoClient, *, full: bool = False
) -> list[JobDryRunCheck]:
    """Check the first rows of every file each dataset's training would read, or with *full*, every row."""
    files = AsyncFilesClient.from_client(async_sdk)
    return [await _check_fileset(files, dataset, workspace, full=full) for dataset in datasets]


def check_local_dataset_rows(dataset: DatasetRows, path: Path, *, full: bool = False) -> JobDryRunCheck:
    """Check a local file or directory the way :func:`check_dataset_rows` checks a fileset."""
    try:
        # Upload puts a lone file at the fileset root under its own name, so training picks it by that name.
        if path.is_file():
            listing = {path.name: path}
        else:
            listing = {f.relative_to(path).as_posix(): f for f in path.rglob("*") if f.is_file()}
        selected = {rel: listing[rel] for rel in dataset.select_files(sorted(listing))}
        for file in selected.values():
            with file.open("rb") as handle:
                if full:
                    dataset.validate_lines(_lines(lambda: handle.read(_CHUNK_BYTES) or None, str(file)), str(file))
                    continue
                head = handle.read(_MAX_BYTES)
            _validate_head(dataset, head, at_end=len(head) == file.stat().st_size, source=str(file))
    except Exception as e:
        return JobDryRunCheck(name=dataset.name, passed=False, message=str(e))
    return _passed(dataset, list(selected), full=full)


async def _check_fileset(
    files: AsyncFilesClient, dataset: DatasetRows, workspace: str, *, full: bool
) -> JobDryRunCheck:
    try:
        ref = FileSetRef.model_validate(dataset.fileset)
        fileset_workspace = ref.workspace or workspace
        prefix = ref.path or ""
        listing = (await files.list_files(workspace=fileset_workspace, name=ref.name)).data().data
        # Paths relative to the reference, as the download step lays them out.
        sizes = {
            file.path.lstrip("/").removeprefix(prefix): file.size
            for file in listing
            if file.path.lstrip("/").startswith(prefix)
        }
        selected = dataset.select_files(sorted(sizes))
        for rel in selected:
            path = f"{prefix}{rel}"
            source = f"{fileset_workspace}/{ref.name}#{path}"
            if full:
                await _validate_whole_file(files, dataset, fileset_workspace, ref.name, path, source=source)
                continue
            head = await _read_head(files, fileset_workspace, ref.name, path, sizes[rel])
            _validate_head(dataset, head, at_end=len(head) >= sizes[rel], source=source)
    except Exception as e:
        return JobDryRunCheck(name=dataset.name, passed=False, message=str(e))
    return _passed(dataset, selected, full=full)


async def _validate_whole_file(
    files: AsyncFilesClient, dataset: DatasetRows, workspace: str, name: str, path: str, *, source: str
) -> None:
    """Validate every row of a file as it streams in from the Files service.

    The validator runs in a worker thread and pulls the next chunk when it needs more lines,
    so memory stays flat, the service keeps answering other requests, and a bad row stops
    the download.
    """
    loop = asyncio.get_running_loop()
    response = await files.download_file(workspace=workspace, name=name, path=path)
    async with response.stream(_CHUNK_BYTES) as chunks:

        def next_chunk() -> bytes | None:
            return asyncio.run_coroutine_threadsafe(_next_chunk(chunks), loop).result()

        await asyncio.to_thread(dataset.validate_lines, _lines(next_chunk, source), source)


async def _next_chunk(chunks: AsyncIterator[bytes]) -> bytes | None:
    return await anext(chunks, None)


def _lines(next_chunk: Callable[[], bytes | None], source: str) -> Iterator[str]:
    """The lines of a file read chunk by chunk, decoded as the validator asks for them."""
    pending, number, rows = b"", 0, 0
    while (chunk := next_chunk()) is not None:
        *complete, pending = (pending + chunk).split(b"\n")
        for line in complete:
            number += 1
            rows += bool(line.strip())
            yield _decode(line, number, source)
    if pending:
        rows += bool(pending.strip())
        yield _decode(pending, number + 1, source)
    if rows == 0:
        raise ValueError(f"{source} is empty.")


def _decode(line: bytes, number: int, source: str) -> str:
    try:
        return line.decode("utf-8")
    except UnicodeDecodeError as e:
        raise ValueError(f"{source} line {number} is not UTF-8 text: {e}") from e


async def _read_head(files: AsyncFilesClient, workspace: str, name: str, path: str, size: int) -> bytes:
    """Read from the start of a file until it holds :data:`ROWS_TO_CHECK` complete nonblank lines."""
    data = b""
    while len(data) < min(size, _MAX_BYTES) and _complete_rows(data) < ROWS_TO_CHECK:
        end = min(len(data) + _CHUNK_BYTES, size)
        response = await files.with_headers({"Range": f"bytes={len(data)}-{end - 1}"}).download_file(
            workspace=workspace, name=name, path=path
        )
        data += await response.read()
    return data


def _complete_rows(data: bytes) -> int:
    """Count the nonblank lines of *data* that end in a newline; training skips blank lines."""
    return sum(1 for line in data.split(b"\n")[:-1] if line.strip())


def _validate_head(dataset: DatasetRows, head: bytes, *, at_end: bool, source: str) -> None:
    """Validate the first :data:`ROWS_TO_CHECK` complete rows in *head*."""
    if at_end and not head.strip():
        raise ValueError(f"{source} is empty.")
    lines = head.split(b"\n")
    if not at_end:
        lines = lines[:-1]  # The last line may be cut off mid-row.
    try:
        text = [line.decode("utf-8") for line in lines]
    except UnicodeDecodeError as e:
        raise ValueError(f"{source} is not UTF-8 text: {e}") from e
    rows, kept = 0, []
    for line in text:
        if rows == ROWS_TO_CHECK:
            break
        kept.append(line)
        rows += bool(line.strip())
    if rows == 0:
        raise ValueError(
            f"The first row of {source} is longer than {_MAX_BYTES // (1024 * 1024)} MiB, "
            "so the dry run cannot check it."
        )
    dataset.validate_lines(kept, source)


def _passed(dataset: DatasetRows, files: list[str], *, full: bool) -> JobDryRunCheck:
    names = ", ".join(files)
    message = (
        f"Every row of {names} matches what training reads."
        if full
        else f"The first {ROWS_TO_CHECK} rows of {names} match what training reads."
    )
    return JobDryRunCheck(name=dataset.name, passed=True, message=message)
