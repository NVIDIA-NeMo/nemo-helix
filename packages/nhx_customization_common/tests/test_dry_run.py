# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import re
from collections.abc import AsyncIterator, Iterable
from contextlib import asynccontextmanager
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from nhx.customization_common import dry_run
from nhx.customization_common.dry_run import (
    DatasetRows,
    check_dataset_rows,
    check_local_dataset_rows,
)


class _FakeFiles:
    """Serves one fileset's files, honoring the Range header like the Files service, or streaming a whole file."""

    def __init__(self, files: dict[str, bytes]) -> None:
        self.files = files
        self.ranges: list[tuple[str, str]] = []
        #: The path of each chunk a whole-file stream sent.
        self.streamed: list[str] = []
        self._range: str | None = None

    def with_headers(self, headers: dict[str, str]) -> "_FakeFiles":
        view = _FakeFiles(self.files)
        view.ranges = self.ranges
        view._range = headers["Range"]
        return view

    async def list_files(self, *, workspace: str, name: str) -> MagicMock:
        assert (workspace, name) == ("ws", "data")
        listing = [SimpleNamespace(path=path, size=len(content)) for path, content in self.files.items()]
        response = MagicMock()
        response.data.return_value = SimpleNamespace(data=listing)
        return response

    async def download_file(self, *, workspace: str, name: str, path: str) -> SimpleNamespace:
        if self._range is None:
            return SimpleNamespace(stream=lambda chunk_size: self._stream(path, chunk_size))
        self.ranges.append((path, self._range))
        match = re.fullmatch(r"bytes=(\d+)-(\d+)", self._range)
        assert match is not None
        start, end = (int(n) for n in match.groups())

        async def read() -> bytes:
            return self.files[path][start : end + 1]

        return SimpleNamespace(read=read)

    @asynccontextmanager
    async def _stream(self, path: str, chunk_size: int) -> AsyncIterator[AsyncIterator[bytes]]:
        content = self.files[path]

        async def chunks() -> AsyncIterator[bytes]:
            for start in range(0, len(content), chunk_size):
                self.streamed.append(path)
                yield content[start : start + chunk_size]

        yield chunks()


def _rows(n: int, row: dict | None = None) -> bytes:
    return b"".join(json.dumps(row or {"text": f"row {i}"}).encode() + b"\n" for i in range(n))


def _require_text(lines: Iterable[str], source: str) -> int:
    """Stands in for a backend validator: it iterates *lines* once, as a stream allows."""
    count = 0
    for number, line in enumerate(lines, start=1):
        if line.strip() and "text" not in json.loads(line):
            raise ValueError(f"{source} line {number} has no 'text'")
        count += 1
    return count


def _dataset(fileset: str = "ws/data") -> DatasetRows:
    return DatasetRows(
        name="training dataset",
        fileset=fileset,
        select_files=lambda paths: [path for path in paths if path.startswith("train")],
        validate_lines=_require_text,
    )


async def _check(files: _FakeFiles, dataset: DatasetRows, *, full: bool = False) -> dry_run.JobDryRunCheck:
    with patch.object(dry_run.AsyncFilesClient, "from_client", return_value=files):
        [check] = await check_dataset_rows([dataset], "ws", MagicMock(), full=full)
    return check


async def test_checks_the_first_rows_of_the_files_training_reads() -> None:
    files = _FakeFiles({"train.jsonl": _rows(5_000), "notes.txt": b"not data"})
    check = await _check(files, _dataset())

    assert check.passed, check.message
    assert check.message == "The first 10 rows of train.jsonl match what training reads."
    # One 64 KB range covers 10 short rows; the rest of the file is never read.
    assert files.ranges == [("train.jsonl", "bytes=0-65535")]


async def test_a_bad_row_fails_with_its_line_number() -> None:
    content = _rows(2) + json.dumps({"prompt": "no text"}).encode() + b"\n" + _rows(20)
    check = await _check(_FakeFiles({"train.jsonl": content}), _dataset())

    assert not check.passed
    assert check.message == "ws/data#train.jsonl line 3 has no 'text'"


async def test_rows_past_the_first_ten_are_not_checked() -> None:
    content = _rows(10) + json.dumps({"bad": True}).encode() + b"\n"
    assert (await _check(_FakeFiles({"train.jsonl": content}), _dataset())).passed


async def test_reads_more_ranges_until_ten_rows_are_complete() -> None:
    long_row = {"text": "x" * 40_000}
    files = _FakeFiles({"train.jsonl": _rows(12, long_row)})
    assert (await _check(files, _dataset())).passed
    assert len(files.ranges) == 7  # ten 40 KB rows need just over 400 KB


async def test_blank_lines_do_not_count_as_rows() -> None:
    content = _rows(1) + b"\n" * 70_000 + json.dumps({"bad": True}).encode() + b"\n"
    check = await _check(_FakeFiles({"train.jsonl": content}), _dataset())

    # Training skips blank lines, so the bad row is the second row and is checked.
    assert (check.passed, check.message) == (False, "ws/data#train.jsonl line 70002 has no 'text'")


def test_a_first_row_past_the_read_limit_fails(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(dry_run, "_MAX_BYTES", 1024 * 1024)
    (tmp_path / "train.jsonl").write_bytes(_rows(2, {"text": "x" * 1024 * 1024}))

    check = check_local_dataset_rows(_dataset(), tmp_path)
    assert (check.passed, check.message) == (
        False,
        f"The first row of {tmp_path}/train.jsonl is longer than 1 MiB, so the dry run cannot check it.",
    )


async def test_a_fileset_path_scopes_the_files() -> None:
    files = _FakeFiles({"sft/train.jsonl": _rows(3), "train.jsonl": b"{broken"})
    check = await _check(files, _dataset("ws/data#sft"))

    assert check.passed, check.message
    assert [path for path, _ in files.ranges] == ["sft/train.jsonl"]


async def test_an_empty_file_or_no_training_files_fail() -> None:
    assert (await _check(_FakeFiles({"train.jsonl": b""}), _dataset())).message == "ws/data#train.jsonl is empty."

    def no_training(paths: list[str]) -> list[str]:
        raise ValueError("No training files found.")

    dataset = DatasetRows("training dataset", "ws/data", no_training, _require_text)
    check = await _check(_FakeFiles({"README.md": b"hi"}), dataset)
    assert (check.passed, check.message) == (False, "No training files found.")


def test_local_dataset_is_checked_the_same_way(tmp_path: Path) -> None:
    (tmp_path / "train.jsonl").write_bytes(_rows(2) + json.dumps({"x": 1}).encode() + b"\n")
    check = check_local_dataset_rows(_dataset(), tmp_path)
    assert (check.passed, check.message) == (False, f"{tmp_path}/train.jsonl line 3 has no 'text'")

    (tmp_path / "train.jsonl").write_bytes(_rows(2))
    assert check_local_dataset_rows(_dataset(), tmp_path / "train.jsonl").passed


def test_a_local_file_is_picked_by_its_name_as_upload_lays_it_out(tmp_path: Path) -> None:
    """Upload puts a lone file at the fileset root under its own name, and training picks it by that name."""

    def select_training(paths: list[str]) -> list[str]:
        if not (selected := [path for path in paths if path.startswith("train")]):
            raise ValueError("No training files found.")
        return selected

    dataset = DatasetRows("training dataset", "ws/data", select_training, _require_text)
    (tmp_path / "rows.json").write_bytes(_rows(2))
    (tmp_path / "train.json").write_bytes(_rows(2) + json.dumps({"x": 1}).encode() + b"\n")

    assert check_local_dataset_rows(dataset, tmp_path / "rows.json").message == "No training files found."
    check = check_local_dataset_rows(dataset, tmp_path / "train.json")
    assert (check.passed, check.message) == (False, f"{tmp_path}/train.json line 3 has no 'text'")


async def test_a_full_dry_run_finds_a_bad_row_past_the_first_ten() -> None:
    files = _FakeFiles({"train.jsonl": _rows(15) + json.dumps({"bad": True}).encode() + b"\n" + _rows(5)})

    assert (await _check(files, _dataset())).passed
    check = await _check(files, _dataset(), full=True)
    assert (check.passed, check.message) == (False, "ws/data#train.jsonl line 16 has no 'text'")


async def test_a_full_dry_run_streams_every_row_instead_of_reading_ranges() -> None:
    files = _FakeFiles({"train.jsonl": _rows(5_000)})

    check = await _check(files, _dataset(), full=True)

    assert (check.passed, check.message) == (True, "Every row of train.jsonl matches what training reads.")
    assert files.ranges == []
    assert files.streamed


async def test_a_bad_row_stops_a_full_dry_run_from_reading_the_rest(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(dry_run, "_CHUNK_BYTES", 64)
    content = json.dumps({"bad": True}).encode() + b"\n" + _rows(1_000)
    files = _FakeFiles({"train.jsonl": content})

    assert not (await _check(files, _dataset(), full=True)).passed
    assert len(files.streamed) < 5 < len(content) // 64


async def test_a_full_dry_run_fails_an_empty_or_non_utf8_file() -> None:
    empty = await _check(_FakeFiles({"train.jsonl": b"\n \n"}), _dataset(), full=True)
    assert empty.message == "ws/data#train.jsonl is empty."

    latin1 = await _check(_FakeFiles({"train.jsonl": _rows(2) + b'{"text": "\xe9"}\n'}), _dataset(), full=True)
    assert latin1.message.startswith("ws/data#train.jsonl line 3 is not UTF-8 text")


def test_a_full_local_dry_run_checks_every_row(tmp_path: Path) -> None:
    (tmp_path / "train.jsonl").write_bytes(_rows(12) + json.dumps({"x": 1}).encode() + b"\n")

    assert check_local_dataset_rows(_dataset(), tmp_path).passed
    check = check_local_dataset_rows(_dataset(), tmp_path, full=True)
    assert (check.passed, check.message) == (False, f"{tmp_path}/train.jsonl line 13 has no 'text'")
