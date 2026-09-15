# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the file-source seam and the per-format readers."""

import io

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from nemo_profiler_plugin.dataset.readers.base import detect_format, get_reader, is_unsupported_data
from nemo_profiler_plugin.source import FileEntry, LocalFileSource

PARQUET_ROWS = [
    {"prompt": "a", "score": 1},
    {"prompt": "b", "score": 2},
    {"prompt": "c", "score": 3},
]


def _write_parquet(path, rows):
    pq.write_table(pa.Table.from_pylist(rows), path)


# --- file source ---------------------------------------------------------------------------------


def test_local_file_source_lists_sorted_with_sizes(tmp_path):
    (tmp_path / "b.jsonl").write_text('{"x": 1}\n')
    (tmp_path / "a.parquet").write_bytes(b"not-real-parquet")  # only listed here, not parsed
    sub = tmp_path / "sub"
    sub.mkdir()
    (sub / "c.jsonl").write_text("{}\n")

    entries = LocalFileSource(tmp_path).list_files()

    assert [e.path for e in entries] == ["a.parquet", "b.jsonl", "sub/c.jsonl"]
    assert all(e.size_bytes > 0 for e in entries)


def test_local_file_source_skips_symlinks_out_of_the_root(tmp_path):
    # `is_file()` resolves a symlink and `open` follows it, so a link planted inside the root would
    # otherwise be read and its column names would reach the profile. A symlinked *directory* is
    # already excluded, because `rglob` declines to descend into one.
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "secret.jsonl").write_text('{"leaked": 1}\n')
    root = tmp_path / "root"
    root.mkdir()
    (root / "train.jsonl").write_text('{"a": 1}\n')
    (root / "escape.jsonl").symlink_to(outside / "secret.jsonl")
    (root / "escape_dir").symlink_to(outside, target_is_directory=True)

    assert [e.path for e in LocalFileSource(root).list_files()] == ["train.jsonl"]


def test_local_file_source_open_refuses_to_leave_the_root(tmp_path):
    # `list_files` declines to *list* a symlink, and enforcing it only there held for exactly as
    # long as every caller came through `list_files`. `open` is the method that reads, so it is
    # where the rule has to hold.
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "secret.jsonl").write_text('{"leaked": 1}\n')
    root = tmp_path / "root"
    root.mkdir()
    (root / "train.jsonl").write_text('{"a": 1}\n')
    (root / "escape.jsonl").symlink_to(outside / "secret.jsonl")
    source = LocalFileSource(root)

    with source.open("train.jsonl") as stream:
        assert stream.read() == b'{"a": 1}\n'  # the ordinary case is untouched

    with pytest.raises(ValueError, match="outside the source root"):
        source.open("escape.jsonl")
    with pytest.raises(ValueError, match="outside the source root"):
        source.open("../outside/secret.jsonl")
    # `root / "/etc/hostname"` is `/etc/hostname`: an absolute path discards the root outright, with
    # no traversal sequence needed, so a `..` check alone would not have caught this one.
    with pytest.raises(ValueError, match="outside the source root"):
        source.open("/etc/hostname")


def test_local_file_source_open_does_not_follow_an_entry_swapped_after_listing(tmp_path):
    # The one case no check made before the open can catch: the entry was a regular file when it was
    # listed and is a symlink by the time it is read.
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "secret.jsonl").write_text('{"leaked": 1}\n')
    root = tmp_path / "root"
    root.mkdir()
    (root / "train.jsonl").write_text('{"a": 1}\n')
    (root / "sibling.jsonl").write_text('{"b": 2}\n')
    source = LocalFileSource(root)
    assert "train.jsonl" in [e.path for e in source.list_files()]

    # Pointing out of the root: refused on containment.
    (root / "train.jsonl").unlink()
    (root / "train.jsonl").symlink_to(outside / "secret.jsonl")
    with pytest.raises(ValueError, match="outside the source root"):
        source.open("train.jsonl")

    # Pointing back *inside* the root, which containment cannot see. O_NOFOLLOW is what refuses it.
    (root / "train.jsonl").unlink()
    (root / "train.jsonl").symlink_to(root / "sibling.jsonl")
    with pytest.raises(OSError):
        source.open("train.jsonl")


def test_local_file_source_open_does_not_leak_a_descriptor_when_wrapping_fails(tmp_path):
    # `os.open` hands back a raw descriptor that nothing owns until `os.fdopen` wraps it. If the
    # wrap raises, the caller never receives a handle and so cannot close it. The profiler opens a
    # file per shard, so leaking one per failure exhausts the process rather than the request.
    import os
    from unittest import mock

    (tmp_path / "a.jsonl").write_text("{}\n")
    source = LocalFileSource(tmp_path)

    def open_fds():
        return len(os.listdir("/proc/self/fd"))

    before = open_fds()
    for _ in range(200):
        with mock.patch("os.fdopen", side_effect=OSError("no memory for a buffer")):
            with pytest.raises(OSError):
                source.open("a.jsonl")
    assert open_fds() == before


def test_local_file_source_open_reads_bytes(tmp_path):
    (tmp_path / "f.jsonl").write_text('{"x": 1}\n')
    with LocalFileSource(tmp_path).open("f.jsonl") as stream:
        assert stream.read() == b'{"x": 1}\n'


def test_local_file_source_rejects_non_directory(tmp_path):
    target = tmp_path / "f"
    target.write_text("x")
    with pytest.raises(NotADirectoryError):
        LocalFileSource(target)


# --- registry ------------------------------------------------------------------------------------


def test_is_unsupported_data_separates_records_from_plumbing():
    # The asymmetry the classification rests on: a false "data" costs one FileError, a false
    # "not data" hides a dataset. So the ambiguous middle -- no extension -- goes to data.
    assert is_unsupported_data("train.csv") is True
    assert is_unsupported_data("train.jsonl.gz") is True  # compressed shard, wrapper suffix only
    assert is_unsupported_data("shard.zst") is True
    assert is_unsupported_data("train") is True  # no extension: assumed data
    # ...and the things that are genuinely not records.
    assert is_unsupported_data("README.md") is False
    assert is_unsupported_data("LICENSE") is False  # no extension, but documentation by name
    assert is_unsupported_data("licence") is False  # matched case-insensitively, either spelling
    assert is_unsupported_data(".gitattributes") is False  # a dotfile reports no suffix at all
    assert is_unsupported_data("dataset_infos.json") is False  # a card, not records
    assert is_unsupported_data("records.json") is True  # plain .json still reads as records
    assert is_unsupported_data("data/dataset_infos.json") is False  # matched on the name, not the path


def test_detect_format_by_extension():
    assert detect_format("data/train-00000-of-00003.parquet") == "parquet"
    assert detect_format("x.jsonl") == "jsonl"
    assert detect_format("x.ndjson") == "jsonl"
    assert detect_format("README.md") is None


def test_get_reader_unknown_format_raises():
    with pytest.raises(KeyError):
        get_reader("arrow")


# --- parquet reader ------------------------------------------------------------------------------


def test_parquet_reader_reads_schema_rows_and_exact_count(tmp_path):
    _write_parquet(tmp_path / "d.parquet", PARQUET_ROWS)
    result = get_reader("parquet").read(LocalFileSource(tmp_path), FileEntry("d.parquet", 0))

    assert result.num_rows == 3  # exact, from the footer
    assert result.examples_scanned == 3
    assert result.rows == PARQUET_ROWS
    assert result.arrow_schema is not None
    assert set(result.arrow_schema.names) == {"prompt", "score"}


def test_parquet_reader_row_cap_bounds_rows_but_keeps_exact_count(tmp_path):
    _write_parquet(tmp_path / "d.parquet", PARQUET_ROWS)
    result = get_reader("parquet").read(LocalFileSource(tmp_path), FileEntry("d.parquet", 0), row_cap=2)

    assert result.num_rows == 3  # footer count is unaffected by sampling
    assert result.examples_scanned == 2
    assert result.rows == PARQUET_ROWS[:2]


def test_parquet_reader_zero_cap_reads_no_rows(tmp_path):
    _write_parquet(tmp_path / "d.parquet", PARQUET_ROWS)
    result = get_reader("parquet").read(LocalFileSource(tmp_path), FileEntry("d.parquet", 0), row_cap=0)

    assert result.rows == []
    assert result.num_rows == 3
    assert result.arrow_schema is not None  # schema is still known without reading rows


# --- magic bytes ---------------------------------------------------------------------------------

# Driven through all three, because a check that guards only one of them guards nothing: the
# pipeline peeks a partition and *discards* the failures, so the reason a file was rejected reaches
# a FileError only if the read path raises it too.
_ENTRY_POINTS = {
    "peek": lambda reader, source, entry: reader.peek(source, entry),
    "read": lambda reader, source, entry: reader.read(source, entry),
    "batches": lambda reader, source, entry: list(reader.batches(source, entry)),
}


def _drive(file_format, entry_point, tmp_path, name):
    return _ENTRY_POINTS[entry_point](get_reader(file_format), LocalFileSource(tmp_path), FileEntry(name, 0))


@pytest.mark.parametrize("entry_point", list(_ENTRY_POINTS))
def test_parquet_reader_rejects_a_file_that_was_never_parquet(tmp_path, entry_point):
    # Format is chosen by extension, so anything named .parquet arrives here — including the HTML
    # error page a failed download saves under the requested filename.
    (tmp_path / "d.parquet").write_bytes(b"<!DOCTYPE html><html><body>404 Not Found</body></html>")

    with pytest.raises(ValueError, match="not a parquet file"):
        _drive("parquet", entry_point, tmp_path, "d.parquet")


@pytest.mark.parametrize("entry_point", list(_ENTRY_POINTS))
def test_parquet_reader_names_truncation_apart_from_the_wrong_format(tmp_path, entry_point):
    # A cut-short upload still opens with PAR1, so only the trailing marker distinguishes it. The two
    # answers point at different problems — a bad name versus a bad transfer — and are worth keeping
    # apart.
    path = tmp_path / "d.parquet"
    _write_parquet(path, PARQUET_ROWS)
    path.write_bytes(path.read_bytes()[:-4])

    with pytest.raises(ValueError, match="truncated parquet file"):
        _drive("parquet", entry_point, tmp_path, "d.parquet")


def test_parquet_reader_rejects_a_file_too_small_for_both_markers(tmp_path):
    # Four bytes of PAR1 and nothing else passes *both* marker checks — the leading and trailing
    # reads land on the same bytes — so the size guard is what rejects it, not the markers.
    (tmp_path / "d.parquet").write_bytes(b"PAR1")

    with pytest.raises(ValueError, match="cannot hold both PAR1 markers"):
        _drive("parquet", "peek", tmp_path, "d.parquet")


# --- jsonl reader --------------------------------------------------------------------------------


def test_jsonl_reader_full_read_is_exact_and_skips_blanks(tmp_path):
    (tmp_path / "d.jsonl").write_text('{"a": 1}\n\n{"a": 2}\n')
    result = get_reader("jsonl").read(LocalFileSource(tmp_path), FileEntry("d.jsonl", 0))

    assert result.rows == [{"a": 1}, {"a": 2}]
    assert result.num_rows == 2  # exact on a full read
    assert result.arrow_schema is None  # jsonl declares no schema


def test_jsonl_reader_row_cap_leaves_count_unknown(tmp_path):
    (tmp_path / "d.jsonl").write_text('{"a": 1}\n{"a": 2}\n{"a": 3}\n')
    result = get_reader("jsonl").read(LocalFileSource(tmp_path), FileEntry("d.jsonl", 0), row_cap=2)

    assert result.rows == [{"a": 1}, {"a": 2}]
    assert result.examples_scanned == 2
    assert result.num_rows is None  # a partial read can't assert the total


def test_jsonl_reader_skips_non_object_lines(tmp_path):
    # A record is a column map; valid JSON that is a scalar or array is not a row.
    (tmp_path / "d.jsonl").write_text('{"a": 1}\n[1, 2, 3]\n42\n"loose"\n{"a": 2}\n')
    result = get_reader("jsonl").read(LocalFileSource(tmp_path), FileEntry("d.jsonl", 0))

    assert result.rows == [{"a": 1}, {"a": 2}]  # stray non-object lines dropped, objects kept
    assert result.num_rows == 2
    # Not a read failure: those lines are not rows of this dataset, so the count stays exact and the
    # file is still exhaustively scanned. Only an *unparseable* line is an error.
    assert result.error is None


def test_jsonl_reader_survives_an_unparseable_line(tmp_path):
    # One truncated line must cost that line, not the file. Dropping the whole file would erase its
    # row count and any column it was the only witness for.
    (tmp_path / "d.jsonl").write_text('{"a": 1}\n{"a": 2\n{"a": 3}\n')
    result = get_reader("jsonl").read(LocalFileSource(tmp_path), FileEntry("d.jsonl", 0))

    assert result.rows == [{"a": 1}, {"a": 3}]  # the readable rows survive
    assert result.examples_scanned == 2
    assert result.error is not None
    assert "line 2" in result.error  # self-describing: which line, and why


def test_jsonl_reader_clean_read_reports_no_error(tmp_path):
    (tmp_path / "d.jsonl").write_text('{"a": 1}\n{"a": 2}\n')
    assert get_reader("jsonl").read(LocalFileSource(tmp_path), FileEntry("d.jsonl", 0)).error is None


@pytest.mark.parametrize("entry_point", list(_ENTRY_POINTS))
@pytest.mark.parametrize(
    ("leading_bytes", "expected"),
    [
        (b"\x1f\x8b\x08\x00\x00\x00\x00\x00", "gzip archive"),  # a .jsonl.gz that lost its suffix
        (b"PAR1\x15\x04\x15\x00\x15\x02", "parquet file"),
        (b"PK\x03\x04\x14\x00\x00\x00", "zip archive"),
        (b"ARROW1\x00\x00\xff\xff\xff\xff", "arrow IPC file"),
    ],
)
def test_jsonl_reader_rejects_a_misnamed_binary_file(tmp_path, leading_bytes, expected, entry_point):
    # Without this the file reports "skipped N unparseable line(s)" — a message that reads like
    # corrupt data and sends the reader looking at the wrong thing.
    (tmp_path / "d.jsonl").write_bytes(leading_bytes + b"\x00" * 64)

    with pytest.raises(ValueError, match=expected):
        _drive("jsonl", entry_point, tmp_path, "d.jsonl")


def test_jsonl_reader_leaves_a_textual_mismatch_to_the_line_parser(tmp_path):
    # A pretty-printed JSON array is the other common misnaming, and is deliberately *not* rejected:
    # it is text, no signature proves it wrong, and a first line that is not an object says nothing
    # about the rest. It stays with `_records`, which reports the lines it could not read rather than
    # discarding a file that may still hold rows — as this one does.
    (tmp_path / "d.jsonl").write_text('[\n  {"a": 1},\n  {"a": 2}\n]\n')

    result = get_reader("jsonl").read(LocalFileSource(tmp_path), FileEntry("d.jsonl", 0))

    assert result.rows == [{"a": 2}]  # the one line that happens to stand alone as an object
    assert result.error is not None and "line 1" in result.error


def test_jsonl_reader_accepts_an_empty_file(tmp_path):
    # Nothing to sniff is not a failed sniff. An empty shard has no rows, which the reader already
    # reports as zero rather than as an error.
    (tmp_path / "d.jsonl").write_bytes(b"")

    result = get_reader("jsonl").read(LocalFileSource(tmp_path), FileEntry("d.jsonl", 0))

    assert result.rows == [] and result.num_rows == 0 and result.error is None


def test_a_failed_reader_import_is_not_latched(monkeypatch):
    # The flag was set *before* the import it guards, so a broken environment became permanent. The
    # real ImportError surfaced exactly once -- inside `_peek_files`' guard, where it is swallowed --
    # and every call after it raised "no reader registered for 'parquet'" instead, so the profile
    # blamed a missing reader for a broken pyarrow and the real cause reached nobody.
    import builtins

    from nemo_profiler_plugin.dataset.readers import base

    monkeypatch.setattr(base, "_builtins_loaded", False)
    real_import = builtins.__import__

    def failing(name, *args, **kwargs):
        if name.startswith("nemo_profiler_plugin.dataset.readers"):
            raise ImportError("simulated broken pyarrow")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", failing)
    with pytest.raises(ImportError, match="simulated broken pyarrow"):
        base.get_reader("parquet")
    # Not latched: the next call tries the import again instead of reporting a missing reader.
    assert base._builtins_loaded is False


def test_parquet_read_honours_the_same_batch_ceiling_as_batches(tmp_path):
    # `read` passed `row_cap` straight to `iter_batches`, so a large cap asked pyarrow for one
    # RecordBatch that size and `to_pylist`-ed it -- materializing the cap, from the contract method
    # whose sibling exists precisely to avoid that.
    from nemo_profiler_plugin.dataset.readers import parquet as parquet_module

    pq.write_table(pa.table({"a": pa.array(range(50))}), tmp_path / "train.parquet")
    entry = FileEntry(path="train.parquet", size_bytes=1)
    source = LocalFileSource(tmp_path)

    seen: list[int] = []
    real = parquet_module.pq.ParquetFile.iter_batches

    def spy(self, *args, batch_size=None, **kwargs):
        seen.append(batch_size)
        return real(self, *args, batch_size=batch_size, **kwargs)

    parquet_module.pq.ParquetFile.iter_batches = spy
    try:
        result = parquet_module.ParquetReader().read(source, entry, row_cap=5_000_000)
    finally:
        parquet_module.pq.ParquetFile.iter_batches = real

    assert seen == [parquet_module._BATCH_ROWS]
    assert len(result.rows) == 50


def test_an_over_long_line_costs_that_line_and_not_the_file(monkeypatch):
    # The malformed input this module already anticipates -- a pretty-printed JSON array saved as
    # `.jsonl` -- is one line as long as the file. Iterating the stream allocated the whole thing as
    # one bytes object, then again for `.strip()`, then the parsed tree, all before discovering it
    # was not a row. `row_cap` cannot help: the cap is checked after the line has been read.
    from nemo_profiler_plugin.dataset.readers import jsonl as jsonl_module

    monkeypatch.setattr(jsonl_module, "_MAX_LINE_BYTES", 64)
    stream = io.BytesIO(b'{"a": 1}\n{"pretty": [' + b"1," * 200 + b'1]}\n{"b": 2}\n')

    out = list(jsonl_module._records(stream))

    # The line is named and skipped; numbering survives it and the rows around it still parse.
    assert [record for record, _ in out] == [{"a": 1}, None, {"b": 2}]
    assert out[1][1] == "line 2: longer than 64 bytes; not read"


def test_a_line_of_exactly_the_bound_is_read_not_discarded(monkeypatch):
    # Reading exactly `_MAX_LINE_BYTES` could not tell "stopped at the limit" from "longer than the
    # limit", so a final line of exactly the bound -- read whole, with nothing left of it -- was
    # thrown away as over-long.
    from nemo_profiler_plugin.dataset.readers import jsonl as jsonl_module

    monkeypatch.setattr(jsonl_module, "_MAX_LINE_BYTES", 16)
    exactly = b'{"bb": 1234567}'  # 15 bytes of JSON
    assert len(exactly) < 16

    for tail, expected in ((exactly, {"bb": 1234567}), (exactly + b"\n", {"bb": 1234567})):
        out = list(jsonl_module._records(io.BytesIO(b'{"a": 1}\n' + tail)))
        assert [record for record, _ in out] == [{"a": 1}, expected]

    # One byte past the bound is still refused.
    out = list(jsonl_module._records(io.BytesIO(b'{"a": 1}\n' + b"x" * 17)))
    assert out[1] == (None, "line 2: longer than 16 bytes; not read")


def test_the_prefix_pair_columns_are_found_when_they_appear_in_a_later_batch():
    # Resolution is bounded by rows examined, not by "the first batch": line-delimited rows are
    # ragged and a column can first appear well into a file.
    from nemo_profiler_plugin.dataset.classify import PrefixPairFold

    shared = "The capital of France is " * 4
    fold = PrefixPairFold()
    fold.update([{"a": 1}])
    fold.update([{"Chosen": shared + "Paris", "Rejected": shared + "Lyon"}])

    assert fold.result().pairs == 1
    assert fold.result().shared == 1
