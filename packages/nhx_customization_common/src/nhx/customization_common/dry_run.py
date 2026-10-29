# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The checks of a customization dry run (create with ``?dry_run=true``).

Besides the execution-profile check every dry run makes, a customization dry run reads
the first rows of each file training would read and checks them with the backend's own
validator. It reads byte ranges from the Files service, so nothing is downloaded to job
storage and nothing is created.
"""

from __future__ import annotations

from collections.abc import Callable
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
    #: Training's own row validator: ``(lines, source)``; raises on a bad row.
    validate_lines: Callable[[list[str], str], object]


async def check_dataset_rows(
    datasets: list[DatasetRows], workspace: str, async_sdk: AsyncNemoClient
) -> list[JobDryRunCheck]:
    """Check the first rows of every file each dataset's training would read."""
    files = AsyncFilesClient.from_client(async_sdk)
    return [await _check_fileset(files, dataset, workspace) for dataset in datasets]


def check_local_dataset_rows(dataset: DatasetRows, path: Path) -> JobDryRunCheck:
    """Check a local file or directory the way :func:`check_dataset_rows` checks a fileset."""
    try:
        if path.is_file():
            selected = {path.name: path}
        else:
            listing = {f.relative_to(path).as_posix(): f for f in path.rglob("*") if f.is_file()}
            selected = {rel: listing[rel] for rel in dataset.select_files(sorted(listing))}
        for rel, file in selected.items():
            with file.open("rb") as handle:
                head = handle.read(_MAX_BYTES)
            _validate_head(dataset, head, at_end=len(head) == file.stat().st_size, source=str(path / rel))
    except Exception as e:
        return JobDryRunCheck(name=dataset.name, passed=False, message=str(e))
    return _passed(dataset, list(selected))


async def _check_fileset(files: AsyncFilesClient, dataset: DatasetRows, workspace: str) -> JobDryRunCheck:
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
            head = await _read_head(files, fileset_workspace, ref.name, path, sizes[rel])
            source = f"{fileset_workspace}/{ref.name}#{path}"
            _validate_head(dataset, head, at_end=len(head) >= sizes[rel], source=source)
    except Exception as e:
        return JobDryRunCheck(name=dataset.name, passed=False, message=str(e))
    return _passed(dataset, selected)


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


def _passed(dataset: DatasetRows, files: list[str]) -> JobDryRunCheck:
    return JobDryRunCheck(
        name=dataset.name,
        passed=True,
        message=f"The first {ROWS_TO_CHECK} rows of {', '.join(files)} match what training reads.",
    )
